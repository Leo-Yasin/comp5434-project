"""第三轮：内部开发选型、冻结一次确认，再拟合最终非线性评分组合。

组号0内部用另一位哈希作80/20选择；组号1不参与树结构/轮数/融合幅度选择。
主指标为均匀替换假设下全体记录风险；显式星级已是输入，代理分数不再作为证据。
"""
import json,time,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import ROOT,write_json
from .lookup_model import fit_residual,residual_design
from .calibration import adjusted_risk
from .regression_boost import ResidualHistogramBoost


def evaluate(p,y,stars,epsilon):
    p=np.clip(p,1,5);nonstar=stars==0
    return {"observed_rmse":float(np.sqrt(np.mean((p-y)**2))),
            "noise_model_risk":adjusted_risk(p,y,epsilon),
            "non_explicit_title_risk":adjusted_risk(p[nonstar],y[nonstar],epsilon),
            "rows":len(y),"nonstar_rows":int(nonstar.sum()),
            "explicit_title_proxy_not_used_for_selection":True}


def initialization(data,fit):
    coefficient=fit_residual(data["base"][fit],data["means"][fit],data["counts"][fit],
                             data["y"][fit],float(data["epsilon"]),context=data["context"][fit])
    x=residual_design(data["base"],data["means"],data["counts"],data["context"])
    return coefficient,data["base"]+x@coefficient


def main():
    out=ROOT/"task2/outputs/round3_20261105"
    data=dict(np.load(out/"development.npz"));test=dict(np.load(out/"test.npz"))
    x,y,groups=data["x"],data["y"],data["groups"];eps=float(data["epsilon"])
    dev=groups%np.uint64(100)==0;inner_valid=dev&((groups//np.uint64(100))%np.uint64(5)==0);fit=dev&~inner_valid;confirm=~dev
    coefficient,initial=initialization(data,fit)
    target=(y-3*eps)/(1-eps)
    baseline=evaluate(initial[inner_valid],y[inner_valid],data["stars"][inner_valid],eps)
    records=[];models={};began=time.perf_counter()
    configs=[{"depth":2,"learning_rate":.05,"min_leaf":700},{"depth":3,"learning_rate":.03,"min_leaf":600}]
    for number,config in enumerate(configs):
        started=time.perf_counter()
        model=ResidualHistogramBoost(rounds=300,l2=800,seed=5434,**config).fit(x[fit],target[fit],initial[fit])
        models[number]=model
        write_json(out/f"inner_model{number}.json",model.to_dict())
        for rounds in [50,100,200,300]:
            p=model.predict(x[inner_valid],initial[inner_valid],rounds)
            for alpha in [.5,1.]:
                mixed=initial[inner_valid]+alpha*(p-initial[inner_valid])
                score=evaluate(mixed,y[inner_valid],data["stars"][inner_valid],eps)
                row={"config":number,**config,"rounds":rounds,"alpha":alpha,"metrics":score}
                records.append(row)
        print("第三轮内部开发",number,"seconds",time.perf_counter()-started,
              "best",min(records,key=lambda r:r["metrics"]["noise_model_risk"]["latent_mse_estimate"]),flush=True)
        write_json(out/"inner_comparison.json",{"linear_baseline":baseline,"candidates":records})
    chosen=min(records,key=lambda r:r["metrics"]["noise_model_risk"]["latent_mse_estimate"])
    # 在查看确认组之前冻结；后续即使确认不佳也不反向搜索该组参数。
    write_json(out/"frozen_selection.json",chosen)
    config=configs[chosen["config"]]
    confirm_coef,confirm_init=initialization(data,dev)
    model=ResidualHistogramBoost(rounds=chosen["rounds"],l2=800,seed=5434,**config).fit(x[dev],target[dev],confirm_init[dev])
    pred=model.predict(x[confirm],confirm_init[confirm]);pred=confirm_init[confirm]+chosen["alpha"]*(pred-confirm_init[confirm])
    confirmation={"baseline":evaluate(confirm_init[confirm],y[confirm],data["stars"][confirm],eps),
                  "candidate":evaluate(pred,y[confirm],data["stars"][confirm],eps),
                  "selection_frozen_before_confirmation":True}
    write_json(out/"confirmation.json",confirmation)
    print("第三轮冻结确认",json.dumps(confirmation,ensure_ascii=False),flush=True)
    pd.DataFrame({"id":data["id"][confirm],"rating":y[confirm],"prediction":np.clip(pred,1,5),
                  "baseline":np.clip(confirm_init[confirm],1,5)}).to_csv(out/"confirmation_predictions.csv",index=False)
    all_fit=np.ones(len(y),dtype=bool);final_coef,final_init=initialization(data,all_fit)
    started=time.perf_counter()
    final=ResidualHistogramBoost(rounds=chosen["rounds"],l2=800,seed=5434,**config).fit(x,target,final_init)
    full_seconds=time.perf_counter()-started
    model_dir=out/"model";model_dir.mkdir(exist_ok=False);support=model_dir/"support_model";support.mkdir()
    # 包内复制所有依赖，避免依赖绝对训练目录。每个文件显式复制，不改旧模型。
    previous=ROOT/"task2/outputs/round2_20261105/model_context"
    (support/"base_model").mkdir()
    for file in previous.iterdir():
        if file.is_file():shutil.copy2(file,support/file.name)
    for file in (previous/"base_model").iterdir():
        if file.is_file():shutil.copy2(file,support/"base_model"/file.name)
    np.savez_compressed(model_dir/"linear.npz",coefficient=final_coef)
    write_json(model_dir/"booster.json",final.to_dict())
    write_json(model_dir/"model.json",{"schema_version":1,"support":"support_model","alpha":chosen["alpha"],
        "features":json.loads((out/"features.json").read_text(encoding="utf-8"))["feature_names"],
        "selected_config":chosen,"full_fit_seconds":full_seconds,"epsilon":eps,
        "selection_did_not_use_title_proxy":True})
    tinitial=test["base"]+residual_design(test["base"],test["means"],test["counts"],test["context"])@final_coef
    tp=final.predict(test["x"],tinitial);tp=np.clip(tinitial+chosen["alpha"]*(tp-tinitial),1,5)
    pd.DataFrame({"id":test["id"],"rating":tp}).to_csv(out/"submission.csv",index=False,float_format="%.10f")
    write_json(out/"timing.json",{"development_and_confirmation_plus_final_seconds":time.perf_counter()-began,
                                  "incremental_final_boost_fit_seconds":full_seconds,
                                  "does_not_include_inherited_word_model_training":True})


if __name__=="__main__":main()
