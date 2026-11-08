"""第四轮对照：继续非线性残差 vs 新增稳健词评分，内部选择、冻结确认。

已有0.60134模型作为继承模型；内部开发使用其原内部训练版本，而非已见确认标签的全量版本。
新候选不会根据Kaggle分数拟合。所有损失与风险的噪声假设同前，不使用标题代理挑选。
"""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import ROOT,write_json
from .round3_train import initialization,evaluate
from .lookup_model import residual_design
from .regression_boost import ResidualHistogramBoost
from .round4_features import refinement_matrix


def old_predict(x,initial,model,alpha):
    return initial+alpha*(model.predict(x,initial)-initial)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["develop","final"],default="develop")
    args=parser.parse_args();out=ROOT/"task2/outputs/round4_20261105";prior=ROOT/"task2/outputs/round3_20261105"
    data=dict(np.load(prior/"development.npz"));test=dict(np.load(prior/"test.npz"));x,y,g=data["x"],data["y"],data["groups"]
    eps=float(data["epsilon"]);target=(y-3*eps)/(1-eps)
    dev=g%np.uint64(100)==0;inner_check=dev&((g//np.uint64(100))%np.uint64(5)==0);fit=dev&~inner_check;confirm=~dev
    auxiliary=np.load(out/"auxiliary_split.npz")["prediction"]
    old_description=json.loads((prior/"model/model.json").read_text(encoding="utf-8"));old_config=old_description["selected_config"]
    if args.stage=="develop":
        began=time.perf_counter();coef,linear=initialization(data,fit)
        cached=ResidualHistogramBoost.from_dict(json.loads((prior/f"inner_model{old_config['config']}.json").read_text(encoding="utf-8")))
        inherited=old_predict(x,linear,cached,old_description["alpha"])
        baseline=evaluate(inherited[inner_check],y[inner_check],data["stars"][inner_check],eps)
        records=[]
        for use_aux in [False,True]:
            features=refinement_matrix(x,inherited,auxiliary,eps,use_aux)
            model=ResidualHistogramBoost(rounds=200,depth=2,learning_rate=.04,min_leaf=700,l2=800,seed=5434).fit(features[fit],target[fit],inherited[fit])
            write_json(out/f"inner_aux{use_aux}.json",model.to_dict())
            for rounds in [50,100,200]:
                p=model.predict(features[inner_check],inherited[inner_check],rounds)
                for alpha in [.5,1.]:
                    pred=inherited[inner_check]+alpha*(p-inherited[inner_check])
                    records.append({"use_auxiliary":use_aux,"rounds":rounds,"alpha":alpha,
                                    "metrics":evaluate(pred,y[inner_check],data["stars"][inner_check],eps)})
            print("第四轮内部对照",use_aux,"best",min(records,key=lambda r:r["metrics"]["noise_model_risk"]["latent_mse_estimate"]),flush=True)
        chosen=min(records,key=lambda r:r["metrics"]["noise_model_risk"]["latent_mse_estimate"])
        write_json(out/"inner_comparison.json",{"inherited_baseline":baseline,"candidates":records})
        write_json(out/"frozen_selection.json",chosen)
        # 冻结后首次检查组号1；先仅在组号0重建前轮非线性模型。
        ccoef,clin=initialization(data,dev)
        inherited_model=ResidualHistogramBoost(rounds=old_config["rounds"],depth=old_config["depth"],
            learning_rate=old_config["learning_rate"],min_leaf=old_config["min_leaf"],l2=800,seed=5434).fit(x[dev],target[dev],clin[dev])
        inherited_confirm=old_predict(x,clin,inherited_model,old_description["alpha"])
        features=refinement_matrix(x,inherited_confirm,auxiliary,eps,chosen["use_auxiliary"])
        model=ResidualHistogramBoost(rounds=chosen["rounds"],depth=2,learning_rate=.04,min_leaf=700,l2=800,seed=5434).fit(features[dev],target[dev],inherited_confirm[dev])
        p=model.predict(features[confirm],inherited_confirm[confirm]);p=inherited_confirm[confirm]+chosen["alpha"]*(p-inherited_confirm[confirm])
        result={"inherited":evaluate(inherited_confirm[confirm],y[confirm],data["stars"][confirm],eps),
                "candidate":evaluate(p,y[confirm],data["stars"][confirm],eps),
                "selection_frozen_before_confirmation":True,"development_seconds":time.perf_counter()-began}
        write_json(out/"confirmation.json",result);print("第四轮确认",json.dumps(result,ensure_ascii=False),flush=True)
        pd.DataFrame({"id":data["id"][confirm],"rating":y[confirm],"baseline":np.clip(inherited_confirm[confirm],1,5),
                      "prediction":np.clip(p,1,5)}).to_csv(out/"confirmation_predictions.csv",index=False)
        return
    chosen=json.loads((out/"frozen_selection.json").read_text(encoding="utf-8"))
    old_booster=ResidualHistogramBoost.from_dict(json.loads((prior/"model/booster.json").read_text(encoding="utf-8")))
    coefficient=np.load(prior/"model/linear.npz")["coefficient"]
    linear=data["base"]+residual_design(data["base"],data["means"],data["counts"],data["context"])@coefficient
    inherited=old_predict(x,linear,old_booster,old_description["alpha"])
    features=refinement_matrix(x,inherited,auxiliary,eps,chosen["use_auxiliary"])
    began=time.perf_counter()
    model=ResidualHistogramBoost(rounds=chosen["rounds"],depth=2,learning_rate=.04,min_leaf=700,l2=800,seed=5434).fit(features,target,inherited)
    seconds=time.perf_counter()-began
    package=out/"model";package.mkdir(exist_ok=False);saved=package/"prior_model";saved.mkdir()
    # 明确枚举复制继承包，旧模型始终只读。没有执行覆盖或移动原模型。
    for source in (prior/"model").rglob("*"):
        relative=source.relative_to(prior/"model");destination=saved/relative
        if source.is_dir():destination.mkdir(exist_ok=True)
        else:shutil.copy2(source,destination)
    if chosen["use_auxiliary"]:shutil.copy2(out/"robust_full/weights.npz",package/"auxiliary.npz")
    write_json(package/"booster.json",model.to_dict());write_json(package/"model.json",{
        "schema_version":1,"prior":"prior_model","chosen":chosen,"incremental_fit_seconds":seconds,
        "additional_depth":2,"additional_learning_rate":.04,"additional_min_leaf":700,"additional_l2":800})
    test_linear=test["base"]+residual_design(test["base"],test["means"],test["counts"],test["context"])@coefficient
    test_prior=old_predict(test["x"],test_linear,old_booster,old_description["alpha"])
    test_aux=(np.load(out/"auxiliary_full.npz")["prediction"] if chosen["use_auxiliary"] else np.zeros(len(test_prior)))
    full_epsilon=json.loads((prior/"model/support_model/model.json").read_text(encoding="utf-8"))["epsilon_full"]
    tx=refinement_matrix(test["x"],test_prior,test_aux,full_epsilon,chosen["use_auxiliary"])
    p=model.predict(tx,test_prior);p=np.clip(test_prior+chosen["alpha"]*(p-test_prior),1,5)
    pd.DataFrame({"id":test["id"],"rating":p}).to_csv(out/"submission.csv",index=False,float_format="%.10f")


if __name__=="__main__":main()
