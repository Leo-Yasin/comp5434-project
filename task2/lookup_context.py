"""检查用户/商品历史上下文是否进一步改善收缩残差。

复用原评分统计，验证行统计只来自外层训练，测试行来自全量训练。
新增两个冷启动-基预测交互，不增加文本缓存、不进行新的 Kaggle 参数搜索。
"""
from pathlib import Path
import gc,json,shutil
import numpy as np
import pandas as pd
from .audit import ROOT,write_json
from .features import ID_FIELDS
from .target_stats import transform_keys
from .lookup_model import fit_residual,predict_residual,metrics
from .calibration import correct


def clean_context(numeric,epsilon):
    context=np.array(numeric,dtype=float,copy=True)
    # 原统计均值已减去 3，逆均匀替换通道只需除以衰减系数。
    context[:,[0,3,6]]/=(1-epsilon)
    return context


def context_from_frame(frame,base_model):
    description=json.loads((base_model/"model.json").read_text(encoding="utf-8"))
    epsilon=description["score_calibration"]["epsilon"]
    columns=[]
    for key in ID_FIELDS:
        keys=pd.util.hash_pandas_object(frame[key].fillna("").astype(str),index=False).to_numpy(dtype=np.uint64)
        with np.load(base_model/f"{key}.npz") as state:columns.append(transform_keys(keys,state))
    return clean_context(np.concatenate(columns,axis=1),epsilon)


def main():
    out=ROOT/"task2/outputs/round2_20261105"
    info=json.loads((out/"evaluation.json").read_text(encoding="utf-8"))
    description=json.loads((out/"model/model.json").read_text(encoding="utf-8"));eps=description["epsilon_split"]
    cache=json.loads((ROOT/"task2/cache/manifest.json").read_text(encoding="utf-8"))
    contexts=[];groups=[]
    for part in cache["parts"]:
        group=np.load(part["prefix"]+".meta.npz")["groups"];valid=group%np.uint64(100)<2
        contexts.append(np.load(part["prefix"]+".split.numeric.npy")[valid]);groups.append(group[valid])
    context=clean_context(np.concatenate(contexts),eps);groups=np.concatenate(groups)
    test_context=clean_context(np.load(ROOT/"task2/cache/test.full.numeric.npy"),description["epsilon_full"])
    old=pd.read_csv(ROOT/"task2/outputs/optimization_20261105/converged/hybrid/validation_predictions.csv")
    base=correct(old.prediction.to_numpy(),eps);y=old.rating.to_numpy();dev=groups%np.uint64(100)==0
    stars=np.load(ROOT/"task2/outputs/optimization_20261105/explicit_title_stars.npy")[old.id.to_numpy(dtype=int)]
    data=np.load(out/"lookup_values.npz");means,counts=data["validation_means"],data["validation_counts"]
    oof=np.empty(len(base));records=[]
    for name,fit,check in [("develop0_confirm1",dev,~dev),("develop1_confirm0",~dev,dev)]:
        coefficient=fit_residual(base[fit],means[fit],counts[fit],y[fit],eps,context=context[fit])
        oof[check]=predict_residual(base[check],means[check],counts[check],coefficient,context=context[check])
        records.append({"split":name,"candidate":metrics(oof[check],y[check],stars[check],eps)})
    result={"confirmation":records,"overall":metrics(oof,y,stars,eps),
            "prior_simple_result":info["overall_cross_fitted"]["candidate"],"new_kaggle_score":None}
    write_json(out/"context_evaluation.json",result)
    pd.DataFrame({"id":old.id,"rating":y,"prediction":oof}).to_csv(out/"context_validation_predictions.csv",index=False)
    final=fit_residual(base,means,counts,y,eps,context=context)
    model=out/"model_context";model.mkdir(exist_ok=False);(model/"base_model").mkdir()
    for source in (out/"model").iterdir():
        if source.is_file():shutil.copy2(source,model/source.name)
    for source in (out/"model/base_model").iterdir():
        if source.is_file():shutil.copy2(source,model/"base_model"/source.name)
    description["use_context"]=True;write_json(model/"model.json",description)
    np.savez_compressed(model/"residual.npz",coefficient=final)
    previous=pd.read_csv(ROOT/"submissions/20261105/task2_optimized_calibrated.csv")
    p=predict_residual(previous.rating.to_numpy(),data["test_means"],data["test_counts"],final,context=test_context)
    pd.DataFrame({"id":previous.id,"rating":p}).to_csv(out/"submission_context.csv",index=False,float_format="%.10f")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
