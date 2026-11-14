"""本地第七轮：历史后验去噪、固定内部选型、一次冻结确认与最终预测。

每阶段独立拟合文本先验；验证查询的标签始终排除于历史归约。
原模型头也按阶段读取，不能拿已见确认评分的final头充当确认基线。
"""
import argparse
import gc
import json
import time
import numpy as np
import pandas as pd
from .audit import ROOT, write_json
from .history_prepare import OUT
from .posterior_history import FIELDS, text_design, prior_probability, posterior_values, add_values, history_design
from .noise_softmax import fit_channel
from .embedding_features import development, stage_features, OUT as PREVIOUS
from .embedding_heads import predict_head, fit_ridge, ridge_predict
from .regression_boost import ResidualHistogramBoost
from .round3_train import evaluate


def masks(data):
    groups=data["groups"];dev=groups%np.uint64(100)==0
    check=dev&((groups//np.uint64(100))%np.uint64(5)==0)
    return {"inner":dev&~check,"confirm":dev,"final":np.ones(len(groups),dtype=bool)},check,~dev


def parent_predictions(data,index,logits,embedding,stage):
    x,parent=stage_features(index,logits,embedding,stage)
    name={"inner":"inner_mlp_12.npz","confirm":"confirm_head.npz","final":"final_head.npz"}[stage]
    description=json.loads((PREVIOUS/"frozen_selection.json").read_text(encoding="utf-8"))
    prediction=predict_head(x,parent,logits,dict(np.load(PREVIOUS/name)),description)
    return x[:,:60].copy(),prediction


def build(stage):
    """按阶段构建后验历史。stage=final才允许全部历史评分进入测试特征。"""
    if (OUT/f"features_{stage}.npz").exists():raise FileExistsError("阶段特征已存在，禁止覆盖。")
    started=time.perf_counter();data,index,logits,embedding=development()
    fits,check,confirm=masks(data);fit=fits[stage]
    meta=dict(np.load(OUT/"metadata.npz"));all_ids=meta["id"]
    selected=data["id"].astype(int);all_stars=np.load(ROOT/"task2/outputs/optimization_20261105/explicit_title_stars.npy",mmap_mode="r")
    scores=np.load(OUT/"word_split.npy",mmap_mode="r")
    design=text_design(scores[selected],all_stars[selected])
    denoiser,report=fit_channel(design,np.zeros((len(design),5)),data["y"],fit,float(data["epsilon"]),ridge=50.,iterations=250)
    np.savez_compressed(OUT/f"{stage}_denoiser.npz",**denoiser)
    write_json(OUT/f"{stage}_denoiser.json",report)
    basic,parent=parent_predictions(data,index,logits,embedding,stage)
    test_meta=dict(np.load(ROOT/"task2/cache/test.meta.npz"))
    queries={k:meta[k][selected] for k in FIELDS}
    states={}
    for field in FIELDS:
        keys=np.unique(np.concatenate([queries[field],test_meta[field]]))
        states[field]={"keys":keys,"sums":np.zeros((len(keys),11))}
    if stage=="final": scores=np.load(OUT/"word_full.npy",mmap_mode="r")
    # 所有外层验证行同时排除；不能只排除当前确认子组。
    excluded=np.zeros(len(all_ids),dtype=bool)
    original_ids=np.load(ROOT/"task2/outputs/round3_20261105/development.npz")["id"]
    excluded[original_ids]=True
    included=0
    for first in range(0,len(all_ids),100000):
        last=min(first+100000,len(all_ids));ids=all_ids[first:last]
        available=np.ones(last-first,dtype=bool) if stage=="final" else ~excluded[ids]
        # 仅计算至少有一个被请求身份的行，等价于全量归约后裁剪表。
        relevant=np.zeros(last-first,dtype=bool)
        for field,state in states.items():
            keys=meta[field][first:last];at=np.searchsorted(state["keys"],keys);safe=np.minimum(at,len(state["keys"])-1)
            relevant|=(at<len(state["keys"]))&(state["keys"][safe]==keys)
        at=np.flatnonzero(available&relevant)+first;included+=len(at)
        if len(at):
            values=posterior_values(prior_probability(text_design(scores[at],all_stars[all_ids[at]]),denoiser),meta["rating"][at],float(data["epsilon"]))
            for field,state in states.items():add_values(state["keys"],state["sums"],meta[field][at],values,np.ones(len(at),dtype=bool))
        if first%1000000==0:print("后验历史归约",stage,first,"seconds",time.perf_counter()-started,flush=True)
    history=history_design(queries,states,parent)
    x=np.column_stack([basic,parent,history]).astype(np.float32)
    if stage=="final":
        test=np.load(PREVIOUS/"test.npz");te=np.load(PREVIOUS/"test.embedding.npy",mmap_mode="r")
        tbase,tparent=parent_predictions(None,None,test["logits"],te,"final")
        np.testing.assert_allclose(tparent,pd.read_csv(PREVIOUS/"submission.csv").rating.to_numpy(),atol=6e-10,rtol=0)
        tx=np.column_stack([tbase,tparent,history_design(test_meta,states,tparent)]).astype(np.float32)
        np.savez_compressed(OUT/"test_features.npz",id=test["id"],x=tx,parent=tparent)
        for field,state in states.items():np.savez_compressed(OUT/f"full.{field}.npz",**state)
    else:
        # 状态只用于审计；原验证行确实未被计数，而不是仅在说明中声明排除。
        assert not np.any(~excluded[original_ids])
    np.savez_compressed(OUT/f"features_{stage}.npz",id=data["id"],x=x,parent=parent,fit=fit)
    write_json(OUT/f"build_{stage}.json",{"seconds":time.perf_counter()-started,"source_rows":len(all_ids),
        "relevant_fit_rows":included,"excluded_outer_validation":0 if stage=="final" else len(original_ids),
        "query_rows":len(selected),"features":x.shape[1],"history_columns":history.shape[1],
        "denoiser_fit_rows":int(fit.sum()),"no_bert_weight_update":True})
    print("历史特征完成",stage,x.shape,"seconds",time.perf_counter()-started,flush=True)


def fit_model(x,parent,data,fit,choice):
    if choice["family"]=="ridge":
        state,report=fit_ridge(x,parent,data["y"],fit,float(data["epsilon"]),choice["ridge"])
        return state,report
    target=(data["y"]-3*float(data["epsilon"]))/ (1-float(data["epsilon"]))
    began=time.perf_counter()
    state=ResidualHistogramBoost(rounds=choice["rounds"],depth=choice["depth"],learning_rate=.035,
        min_leaf=400,l2=800,bins=48,seed=5441).fit(x[fit],target[fit],parent[fit])
    return state,{"seconds":time.perf_counter()-began,"training_rows":int(fit.sum())}


def prediction(state,x,parent,choice):
    changed=ridge_predict(x,parent,state) if choice["family"]=="ridge" else state.predict(x,parent,choice["rounds"])
    return np.clip(parent+choice["alpha"]*(changed-parent),1,5)


def save_model(state,path,choice):
    if choice["family"]=="ridge":np.savez_compressed(path.with_suffix(".npz"),**state)
    else:write_json(path.with_suffix(".json"),state.to_dict())


def develop():
    if (OUT/"frozen_selection.json").exists():raise FileExistsError("已冻结，不重选。")
    data,*_=development();fits,check,confirm=masks(data)
    inner=np.load(OUT/"features_inner.npz");x,parent=inner["x"],inner["parent"];fit=fits["inner"]
    def score(p,mask):return evaluate(p[mask],data["y"][mask],data["stars"][mask],float(data["epsilon"]))
    baseline=score(parent,check);records=[]
    for family,parameter in [("ridge",100.),("ridge",1000.),("tree",2),("tree",3)]:
        choice={"family":family,"ridge":parameter} if family=="ridge" else {"family":family,"rounds":300,"depth":parameter}
        model,report=fit_model(x,parent,data,fit,choice)
        save_model(model,OUT/f"inner_{family}_{parameter}",choice)
        for rounds in ([100,300] if family=="tree" else [0]):
            for alpha in [.5,1.]:
                c={**choice,"alpha":alpha}
                if family=="tree":c["rounds"]=rounds
                p=prediction(model,x,parent,c);records.append({**c,"metrics":score(p,check)})
        print("历史模型内部比较",choice,report,flush=True)
        write_json(OUT/"inner_comparison.json",{"baseline":baseline,"candidates":records})
    chosen=min(records,key=lambda row:row["metrics"]["noise_model_risk"]["latent_mse_estimate"])
    chosen["internal_improved"]=chosen["metrics"]["noise_model_risk"]["latent_mse_estimate"]<baseline["noise_model_risk"]["latent_mse_estimate"]
    write_json(OUT/"frozen_selection.json",chosen)
    print("历史模型已冻结",chosen,flush=True)
    del x;gc.collect()


def confirm_and_final(stage):
    data,*_=development();fits,check,confirm=masks(data)
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    cache=np.load(OUT/f"features_{stage}.npz");x,parent=cache["x"],cache["parent"]
    state,report=fit_model(x,parent,data,fits[stage],choice);save_model(state,OUT/f"{stage}_head",choice)
    p=prediction(state,x,parent,choice);write_json(OUT/f"{stage}_training.json",report)
    if stage=="confirm":
        # 冻结配置的同学习器去历史消融，仅解释贡献，不能用结果替换已经冻结的模型。
        ablated,ablation_time=fit_model(x[:,:61],parent,data,fits[stage],choice)
        no_history=prediction(ablated,x[:,:61],parent,choice)
        result={"baseline":evaluate(parent[confirm],data["y"][confirm],data["stars"][confirm],float(data["epsilon"])),
            "candidate":evaluate(p[confirm],data["y"][confirm],data["stars"][confirm],float(data["epsilon"])),
            "without_history":evaluate(no_history[confirm],data["y"][confirm],data["stars"][confirm],float(data["epsilon"])),
            "frozen_before_confirmation":True,"historical_validation_reuse":True}
        pd.DataFrame({"id":data["id"][confirm],"rating":data["y"][confirm],"baseline":parent[confirm],
            "prediction":p[confirm],"without_history":no_history[confirm],"groups":data["groups"][confirm]}).to_csv(OUT/"confirmation_predictions.csv",index=False)
        write_json(OUT/"confirmation.json",result);print("后验历史冻结确认",result,flush=True)
    else:
        confirmed=json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
        if not choice["internal_improved"] or confirmed["candidate"]["noise_model_risk"]["latent_mse_estimate"]>=confirmed["baseline"]["noise_model_risk"]["latent_mse_estimate"]:
            raise RuntimeError("未通过内部和确认双检查，不发布新候选。")
        test=np.load(OUT/"test_features.npz");tp=prediction(state,test["x"],test["parent"],choice)
        pd.DataFrame({"id":test["id"],"rating":tp}).to_csv(OUT/"submission.csv",index=False,float_format="%.10f")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["inner","confirm","final","develop","evaluate","finish"],required=True)
    args=parser.parse_args()
    if args.stage in ["inner","confirm","final"]:build(args.stage)
    elif args.stage=="develop":develop()
    else:confirm_and_final("confirm" if args.stage=="evaluate" else "final")
