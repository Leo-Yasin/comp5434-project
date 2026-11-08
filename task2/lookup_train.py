"""聚合全部评论统计，在开发/确认拆分中检验新增残差，再准备全量模型。

大数据任务的键由三个真实 Dask 进程生成；这里按字段顺序归约，限制峰值内存。
不再重建约 3.6 GiB 的词特征，复用上一轮校正模型，减少离线成本。
"""
import gc,json,time,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import ROOT,write_json,sha256
from .lookup_features import KEYS
from .lookup_model import lookup,fit_residual,predict_residual,metrics
from .calibration import correct,estimate_rate


def main():
    out=ROOT/"task2/outputs/round2_20261105"
    manifest=json.loads((out/"keys_manifest.json").read_text(encoding="utf-8"));parts=manifest["parts"]
    groups=np.concatenate([np.load(p["path"])["groups"] for p in parts])
    labels=np.concatenate([np.load(p["path"])["rating"] for p in parts]).astype(float)
    valid=groups%np.uint64(100)<2;fit=~valid
    indices=np.flatnonzero(valid);development=groups[valid]%np.uint64(100)==0
    validation=pd.read_csv(ROOT/"task2/outputs/optimization_20261105/converged/hybrid/validation_predictions.csv")
    if not np.array_equal(validation.id.to_numpy(),indices):raise ValueError("基模型验证 id 错位。")
    diagnosis=json.loads((ROOT/"task2/outputs/optimization_20261105/title_audit_split.json").read_text(encoding="utf-8"))
    eps_split=estimate_rate(diagnosis["fit_table"]);eps_full=estimate_rate(diagnosis["title_star_vs_observed_rating"])
    base=correct(validation.prediction.to_numpy(),eps_split);stars=np.load(ROOT/"task2/outputs/optimization_20261105/explicit_title_stars.npy")[valid]
    stats=out/"stats";stats.mkdir(exist_ok=False)
    vmean=np.empty((valid.sum(),len(KEYS)));vcount=np.empty_like(vmean)
    test=dict(np.load(out/"test_keys.npz"));tmean=np.empty((len(test["id"]),len(KEYS)));tcount=np.empty_like(tmean)
    coverage={};times={}
    for j,key in enumerate(KEYS):
        began=time.perf_counter()
        keys=np.concatenate([np.load(p["path"])[key] for p in parts])
        usable=np.concatenate([np.load(p["path"])["valid_"+key] for p in parts]) if key in ["title","comment","text_pair"] else np.ones(len(labels),dtype=bool)
        unique,inverse=np.unique(keys,return_inverse=True)
        for mode,mask,epsilon in [("split",fit&usable,eps_split),("full",usable,eps_full)]:
            counts=np.bincount(inverse[mask],minlength=len(unique)).astype(np.int32)
            sums=np.bincount(inverse[mask],weights=labels[mask],minlength=len(unique))
            state={"keys":unique,"counts":counts,"sums":sums,"epsilon":np.array(epsilon)}
            np.savez_compressed(stats/f"{mode}.{key}.npz",**state)
            if mode=="split":
                vmean[:,j],vcount[:,j]=lookup(keys[valid],state,usable[valid])
            else:
                tmean[:,j],tcount[:,j]=lookup(test[key],state,test.get("valid_"+key))
        coverage[key]={"validation_seen_fraction":float((vcount[:,j]>0).mean()),
                       "test_seen_fraction":float((tcount[:,j]>0).mean()),
                       "test_count_median":float(np.median(tcount[:,j])),"test_max_count":float(tcount[:,j].max())}
        times[key]=time.perf_counter()-began
        print("新增统计",key,coverage[key],flush=True)
        del keys,inverse,unique,state,counts,sums,usable;gc.collect()
    np.savez_compressed(out/"lookup_values.npz",validation_means=vmean,validation_counts=vcount,
                        test_means=tmean,test_counts=tcount)
    y=validation.rating.to_numpy();records=[]
    # 固定 ridge 与收缩尺度，只用一半开发另一半确认，再对称检查。
    # 不依赖标题代理来选择参数，按全体记录的噪声模型风险评估。
    meta_oof=np.empty(len(base))
    for name,train_mask,check_mask in [("develop0_confirm1",development,~development),("develop1_confirm0",~development,development)]:
        coefficient=fit_residual(base[train_mask],vmean[train_mask],vcount[train_mask],y[train_mask],eps_split)
        p=predict_residual(base[check_mask],vmean[check_mask],vcount[check_mask],coefficient)
        meta_oof[check_mask]=p
        records.append({"split":name,"baseline":metrics(base[check_mask],y[check_mask],stars[check_mask],eps_split),
                        "candidate":metrics(p,y[check_mask],stars[check_mask],eps_split)})
    overall={"baseline":metrics(base,y,stars,eps_split),"candidate":metrics(meta_oof,y,stars,eps_split)}
    result={"confirmation":records,"overall_cross_fitted":overall,"coverage":coverage,
            "statistics_seconds":times,"ridge":100,"alphas":[2,10,50],"new_kaggle_score":None}
    write_json(out/"evaluation.json",result)
    pd.DataFrame({"id":indices,"rating":y,"baseline":base,"prediction":meta_oof,
                  "development_half":development.astype(int)}).to_csv(out/"validation_predictions.csv",index=False)
    final_coef=fit_residual(base,vmean,vcount,y,eps_split)
    model=out/"model";model.mkdir(exist_ok=False);base_model=model/"base_model";base_model.mkdir()
    previous=ROOT/"task2/outputs/optimization_20261105/full/model_calibrated"
    for file in previous.iterdir():
        if file.is_file():shutil.copy2(file,base_model/file.name)
    for key in KEYS:shutil.copy2(stats/f"full.{key}.npz",model/f"{key}.npz")
    np.savez_compressed(model/"residual.npz",coefficient=final_coef)
    write_json(model/"model.json",{"schema_version":1,"base":"base_model","keys":KEYS,
        "alphas":[2,10,50],"ridge":100,"epsilon_split":eps_split,"epsilon_full":eps_full,
        "training_method":"cross-fitted residual on original held-out review groups",
        "original_label_noise_assumption_retained":True})
    old=pd.read_csv(ROOT/"submissions/20261105/task2_optimized_calibrated.csv")
    prediction=predict_residual(old.rating.to_numpy(),tmean,tcount,final_coef)
    pd.DataFrame({"id":old.id,"rating":prediction}).to_csv(out/"submission.csv",index=False,float_format="%.10f")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
