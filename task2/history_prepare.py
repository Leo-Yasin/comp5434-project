"""第七轮本地实验准备：流式计算两种已有词模型的纯文本评分。

只读取已有稀疏缓存，不重新解析2GB评论。保存紧凑评分而不保存大矩阵副本。
split词模型不含全部原验证行；full只用于最终测试历史。
"""
import argparse
import gc
import json
import time
import numpy as np
from scipy import sparse
from .audit import ROOT, write_json
from .features import TEXT_DIM

OUT = ROOT/"task2/outputs/round7_local_20261114"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode", choices=["split", "full"], required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/f"word_{args.mode}.npy"
    if target.exists(): raise FileExistsError("禁止覆盖已有评分缓存。")
    manifest=json.loads((ROOT/"task2/cache/manifest.json").read_text(encoding="utf-8"))
    paths=[ROOT/"task2/outputs/optimization_20261105"/("converged/hybrid/best_weights.npz" if args.mode=="split" else "full/hybrid/last_weights.npz"),
           ROOT/f"task2/outputs/round4_20261105/robust_{args.mode}/weights.npz"]
    models=[dict(np.load(path)) for path in paths]
    result=np.lib.format.open_memmap(target,mode="w+",dtype=np.float32,shape=(manifest["rows"],2))
    metadata={k:[] for k in ["id","groups","rating","user_id","prod_id","parent_prod_id"]}
    started=time.perf_counter()
    for part in manifest["parts"]:
        prefix=part["prefix"];first=part["start"];last=first+part["rows"]
        x=sparse.csr_matrix((np.load(prefix+".data.npy",mmap_mode="r"),np.load(prefix+".indices.npy",mmap_mode="r"),
                            np.load(prefix+".indptr.npy",mmap_mode="r")),shape=(part["rows"],TEXT_DIM),copy=False)
        for j,state in enumerate(models):result[first:last,j]=float(state["mean"])+x@state["weights"][:TEXT_DIM]
        if args.mode=="split":
            with np.load(prefix+".meta.npz") as meta:
                for key in metadata:metadata[key].append(meta[key])
        if part["number"]%20==0:print("历史文本评分",args.mode,part["number"],"seconds",time.perf_counter()-started,flush=True)
        del x;gc.collect()
    result.flush()
    if args.mode=="split":
        # 使用实际训练缓存的验证ID，不能用早期audit重新计算后直接替换划分。
        merged={k:np.concatenate(v) for k,v in metadata.items()}
        expected=np.load(ROOT/"task2/outputs/round3_20261105/development.npz")["id"]
        np.testing.assert_array_equal(merged["id"][merged["groups"]%np.uint64(100)<2],expected)
        np.savez_compressed(OUT/"metadata.npz",**merged)
    write_json(OUT/f"word_{args.mode}.json",{"seconds":time.perf_counter()-started,"rows":manifest["rows"],
        "pure_text_only":True,"source_models":[str(p.relative_to(ROOT)) for p in paths]})


if __name__=="__main__":main()
