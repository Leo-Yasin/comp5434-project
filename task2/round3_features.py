"""第三轮低维非线性特征，复用旧词模型，不读取测试标签。

开发特征来自原外层留出行，查找表只含原训练部分评分。测试特征使用全量表。
明确星级标题作为输入后，不再用星级标题代理 RMSE 选择模型。
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy import sparse
from .audit import ROOT,CSV_OPTIONS,write_json
from .features import TEXT_DIM,TITLE_DIM,BODY_DIM
from .lookup_features import KEYS,make_keys
from .lookup_model import lookup
from .lookup_context import clean_context,context_from_frame
from .calibration import correct

STAR_NAMES={"one star":1,"one stars":1,"two stars":2,"three stars":3,"four stars":4,"five stars":5}
FEATURE_NAMES=(["base_calibrated","base_unclipped"]+[f"mean_{k}" for k in KEYS]+
               [f"log_count_{k}" for k in KEYS]+[f"history_{i}" for i in range(9)]+
               [f"basic_{i}" for i in range(8)]+["explicit_title_star"]+
               [f"delta_{k}" for k in KEYS])


def assemble(base,raw,means,counts,context,basic,stars):
    """树可以按支持数学习非线性门控，未知键由零支持数区分。"""
    matrix=np.column_stack([base,raw,means,np.log1p(counts),context,basic,stars,means-base[:,None]])
    if not np.isfinite(matrix).all():raise ValueError("新特征含 NaN 或无穷值。")
    return matrix.astype(np.float32)


def features_from_frame(frame,support):
    """独立推理路径：原始 CSV + 包内参数即可构建全部新特征，不需要缓存。"""
    from .features import build_text
    support=Path(support);base_model=support/"base_model"
    description=json.loads((support/"model.json").read_text(encoding="utf-8"))
    eps=description["epsilon_full"]
    context=context_from_frame(frame,base_model)
    numeric=context.copy();numeric[:,[0,3,6]]*=(1-eps)
    # 原评分统计在最初词模型中以 float32 编码，恢复同一精度保证两条数据流一致。
    numeric=numeric.astype(np.float32)
    text=build_text(frame);state=np.load(base_model/"weights.npz")
    raw_noisy=float(state["mean"])+text@state["weights"][:TEXT_DIM]+numeric@state["weights"][TEXT_DIM:]
    raw=(raw_noisy-3*eps)/(1-eps);base=np.clip(raw,1,5)
    keys=make_keys(frame);means=np.empty((len(frame),len(KEYS)));counts=np.empty_like(means)
    for j,key in enumerate(KEYS):
        with np.load(support/f"{key}.npz") as stats:
            means[:,j],counts[:,j]=lookup(keys[key],stats,keys.get("valid_"+key))
    stars=frame.title.fillna("").str.strip().str.lower().map(STAR_NAMES).fillna(0).to_numpy()
    basic=text[:,TITLE_DIM+BODY_DIM:].toarray()
    return assemble(base,raw,means,counts,context,basic,stars),base,means,counts,context


def main():
    out=ROOT/"task2/outputs/round3_20261105";out.mkdir(exist_ok=False);began=time.perf_counter()
    previous=ROOT/"task2/outputs/round2_20261105"
    description=json.loads((previous/"model_context/model.json").read_text(encoding="utf-8"))
    eps=description["epsilon_split"]
    old=pd.read_csv(ROOT/"task2/outputs/optimization_20261105/converged/hybrid/validation_predictions.csv")
    w=np.load(ROOT/"task2/outputs/optimization_20261105/converged/hybrid/best_weights.npz")
    manifest=json.loads((ROOT/"task2/cache/manifest.json").read_text(encoding="utf-8"))
    groups=[];contexts=[];basics=[];raws=[]
    for part in manifest["parts"]:
        prefix=part["prefix"]
        with np.load(prefix+".meta.npz") as meta:g=meta["groups"]
        valid=g%np.uint64(100)<2;groups.append(g[valid])
        x=sparse.csr_matrix((np.load(prefix+".data.npy",mmap_mode="r"),
                             np.load(prefix+".indices.npy",mmap_mode="r"),
                             np.load(prefix+".indptr.npy",mmap_mode="r")),shape=(len(g),TEXT_DIM),copy=False)[valid]
        numeric=np.load(prefix+".split.numeric.npy",mmap_mode="r")[valid]
        raws.append(float(w["mean"])+x@w["weights"][:TEXT_DIM]+numeric@w["weights"][TEXT_DIM:])
        basics.append(x[:,TITLE_DIM+BODY_DIM:].toarray());contexts.append(numeric)
    noisy=np.concatenate(raws)
    np.testing.assert_allclose(np.clip(noisy,1,5),old.prediction.to_numpy(),rtol=0,atol=1e-10)
    raw=(noisy-3*eps)/(1-eps);base=np.clip(raw,1,5)
    context=clean_context(np.concatenate(contexts),eps);groups=np.concatenate(groups)
    lookup_values=np.load(previous/"lookup_values.npz");means=lookup_values["validation_means"];counts=lookup_values["validation_counts"]
    stars=np.load(ROOT/"task2/outputs/optimization_20261105/explicit_title_stars.npy")[old.id.to_numpy(dtype=int)]
    x=assemble(base,raw,means,counts,context,np.concatenate(basics),stars)
    np.savez_compressed(out/"development.npz",x=x,y=old.rating.to_numpy(),groups=groups,base=base,
                        means=means,counts=counts,context=context,stars=stars,id=old.id.to_numpy(),epsilon=eps)
    test=pd.read_csv(ROOT/"comp-5434-20261-project-task-2/test.csv",**CSV_OPTIONS)
    tx,tbase,tm,tc,tcontext=features_from_frame(test,previous/"model_context")
    np.savez_compressed(out/"test.npz",x=tx,base=tbase,means=tm,counts=tc,context=tcontext,id=test.id.to_numpy())
    write_json(out/"features.json",{"feature_names":FEATURE_NAMES,"columns":x.shape[1],
        "rows":len(x),"seconds":time.perf_counter()-began,
        "explicit_stars_are_input_proxy_not_independent":True,"old_noisy_predictions_match":True})
    print("第三轮特征完成",x.shape,tx.shape,flush=True)


if __name__=="__main__":main()
