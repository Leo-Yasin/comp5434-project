"""从稳健词模型提取原外层验证/测试辅助评分，不将其自动视为干净标签。

树输入可同时看到原尺度、逆通道尺度及与旧词模型的差，学习何时信任新信号。
"""
import argparse,json
import numpy as np
from scipy import sparse
from .audit import ROOT,write_json
from .features import TEXT_DIM


def refinement_matrix(x,prior,auxiliary,epsilon,use_auxiliary=True):
    if not use_auxiliary:return np.column_stack([x,prior]).astype(np.float32)
    previous_noisy=x[:,1]*(1-epsilon)+3*epsilon
    return np.column_stack([x,prior,auxiliary,(auxiliary-3*epsilon)/(1-epsilon),
                             auxiliary-previous_noisy]).astype(np.float32)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["split","full"],default="split")
    args=parser.parse_args();out=ROOT/"task2/outputs/round4_20261105"
    state=np.load(out/("robust_"+args.mode)/"weights.npz");weights=state["weights"];mean=float(state["mean"])
    if args.mode=="split":
        manifest=json.loads((ROOT/"task2/cache/manifest.json").read_text(encoding="utf-8"));predictions=[];ids=[]
        for part in manifest["parts"]:
            prefix=part["prefix"]
            with np.load(prefix+".meta.npz") as meta:
                valid=meta["groups"]%np.uint64(100)<2;ids.append(meta["id"][valid]);n=len(valid)
            x=sparse.csr_matrix((np.load(prefix+".data.npy",mmap_mode="r"),np.load(prefix+".indices.npy",mmap_mode="r"),
                                 np.load(prefix+".indptr.npy",mmap_mode="r")),shape=(n,TEXT_DIM),copy=False)[valid]
            numeric=np.load(prefix+".split.numeric.npy",mmap_mode="r")[valid]
            predictions.append(mean+x@weights[:TEXT_DIM]+numeric@weights[TEXT_DIM:])
        ids=np.concatenate(ids);predictions=np.concatenate(predictions)
        expected=np.load(ROOT/"task2/outputs/round3_20261105/development.npz")["id"]
        if not np.array_equal(ids,expected):raise RuntimeError("稳健辅助验证 id 错位。")
    else:
        x=sparse.load_npz(ROOT/"task2/cache/test.text.npz");numeric=np.load(ROOT/"task2/cache/test.full.numeric.npy")
        predictions=mean+x@weights[:TEXT_DIM]+numeric@weights[TEXT_DIM:]
        ids=np.load(ROOT/"task2/cache/test.meta.npz")["id"]
    np.savez_compressed(out/("auxiliary_"+args.mode+".npz"),id=ids,prediction=predictions)
    print("稳健辅助特征完成",args.mode,len(ids),float(predictions.mean()),flush=True)


if __name__=="__main__":main()
