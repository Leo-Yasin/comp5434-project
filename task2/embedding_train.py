"""第六轮有限内部选型、冻结确认、全量头训练与自包含模型包导出。

只训练冻结表示之后的头；每阶段明确保存计时、优化状态和去CLS消融。
确认不改善就不发布新候选，不用确认标签继续调超参数。
"""
import argparse
import gc
import json
import shutil
import time
import numpy as np
import pandas as pd

from .audit import ROOT,write_json,sha256
from .embedding_features import OUT,OLD,development,stage_features
from .embedding_heads import fit_ridge,fit_linear,fit_mlp,predict_head
from .round3_train import evaluate


def score(data,p,mask):
    return evaluate(p[mask],data["y"][mask],data["stars"][mask],float(data["epsilon"]))


def fit_one(features,parent,logits,data,fit,choice):
    eps=float(data["epsilon"])
    if choice["family"]=="ridge":
        return fit_ridge(features,parent,data["y"],fit,eps,choice["ridge"])
    if choice["family"]=="linear":
        return fit_linear(features,logits,data["y"],fit,eps,choice["ridge"])
    if choice["family"]=="mlp":
        anchor,report=fit_linear(features,logits,data["y"],fit,eps,1000.)
        saved,mlp_report=fit_mlp(features,logits,data["y"],fit,eps,anchor,epochs=choice["epochs"])
        return saved[choice["epochs"]],{"anchor":report,"mlp":mlp_report}
    raise ValueError("未知模型头。")


def develop():
    if (OUT/"frozen_selection.json").exists():raise FileExistsError("本轮选型已冻结。")
    data,index,logits,embedding=development()
    groups=data["groups"];dev=groups%np.uint64(100)==0
    check=dev&((groups//np.uint64(100))%np.uint64(5)==0);fit=dev&~check
    x,parent=stage_features(index,logits,embedding,"inner")
    baseline=score(data,parent,check);records=[];anchor=None
    began=time.perf_counter()
    for family in ["ridge","linear"]:
        for ridge in [100.,1000.]:
            choice={"family":family,"ridge":ridge}
            state,report=fit_one(x,parent,logits,data,fit,choice)
            name=f"inner_{family}_{int(ridge)}"
            np.savez_compressed(OUT/(name+".npz"),**state);write_json(OUT/(name+".json"),report)
            for alpha in [.5,1.]:
                c={**choice,"alpha":alpha};p=predict_head(x,parent,logits,state,c)
                records.append({**c,"metrics":score(data,p,check)})
            if family=="linear" and ridge==1000.:anchor=state
            print("第六轮内部模型",choice,"seconds",report["seconds"],flush=True)
    states,report=fit_mlp(x,logits,data["y"],fit,float(data["epsilon"]),anchor,epochs=12)
    write_json(OUT/"inner_mlp_training.json",report)
    for epochs,state in states.items():
        np.savez_compressed(OUT/f"inner_mlp_{epochs}.npz",**state)
        for alpha in [.5,1.]:
            choice={"family":"mlp","epochs":epochs,"alpha":alpha}
            p=predict_head(x,parent,logits,state,choice)
            records.append({**choice,"metrics":score(data,p,check)})
    chosen=min(records,key=lambda row:row["metrics"]["noise_model_risk"]["latent_mse_estimate"])
    write_json(OUT/"inner_comparison.json",{"parent":baseline,"candidates":records,
        "seconds":time.perf_counter()-began,"no_confirmation_used":True})
    write_json(OUT/"frozen_selection.json",chosen)
    print("第六轮内部冻结",json.dumps(chosen,ensure_ascii=False),flush=True)
    del x;gc.collect()
    # 只有配置冻结后才使用group=1评分；旧第五轮confirm只曾在group=0拟合。
    x,parent=stage_features(index,logits,embedding,"confirm")
    state,report=fit_one(x,parent,logits,data,dev,chosen)
    np.savez_compressed(OUT/"confirm_head.npz",**state);write_json(OUT/"confirm_training.json",report)
    p=predict_head(x,parent,logits,state,chosen)
    # 同训练目标/超参数移除CLS的只读消融，不能反向替换已冻结候选。
    ablated,ablation_report=fit_one(x[:,:60],parent,logits,data,dev,chosen)
    ablation=predict_head(x[:,:60],parent,logits,ablated,chosen)
    result={"parent":score(data,parent,~dev),"candidate":score(data,p,~dev),
        "without_cls":score(data,ablation,~dev),"selection_frozen_before_confirmation":True,
        "historical_validation_reuse_disclosed":True,"external_model_teacher_permission_user_report":True}
    write_json(OUT/"confirmation.json",result);write_json(OUT/"ablation_training.json",ablation_report)
    pd.DataFrame({"id":data["id"][~dev],"rating":data["y"][~dev],"parent":parent[~dev],
        "prediction":p[~dev],"without_cls":ablation[~dev],"groups":groups[~dev]}).to_csv(OUT/"confirmation_predictions.csv",index=False)
    print("第六轮冻结确认",json.dumps(result,ensure_ascii=False),flush=True)


def final():
    if (OUT/"model").exists():raise FileExistsError("第六轮模型已存在，禁止覆盖。")
    result=json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
    if result["candidate"]["noise_model_risk"]["latent_mse_estimate"]>=result["parent"]["noise_model_risk"]["latent_mse_estimate"]:
        raise RuntimeError("冻结确认未改善，不发布新候选。")
    data,index,logits,embedding=development()
    chosen=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    x,parent=stage_features(index,logits,embedding,"final")
    state,report=fit_one(x,parent,logits,data,np.ones(len(x),dtype=bool),chosen)
    np.savez_compressed(OUT/"final_head.npz",**state);write_json(OUT/"final_training.json",report)
    del x;gc.collect()
    test=np.load(OUT/"test.npz");te=np.load(OUT/"test.embedding.npy",mmap_mode="r")
    tx,tp=stage_features(None,test["logits"],te,"final")
    previous=pd.read_csv(ROOT/"submissions/20261105_task2_round5/task2_semantic_noise_fusion.csv")
    if not np.array_equal(previous.id.to_numpy(),test["id"]):raise ValueError("测试id错位。")
    max_difference=float(np.max(np.abs(tp-previous.rating.to_numpy())))
    if max_difference>6e-10:raise ValueError("相同测试输入下第五轮基线发生意外变化。")
    p=predict_head(tx,tp,test["logits"],state,chosen)
    pd.DataFrame({"id":test["id"],"rating":p}).to_csv(OUT/"submission.csv",index=False,float_format="%.10f")
    package=OUT/"model";package.mkdir();support=package/"support";support.mkdir()
    # 只复制真正部署所需的上一轮参数；新包只有一份带CLS输出的网络，避免重复大型权重。
    shutil.copytree(OLD/"model/prior_model",support/"previous_model")
    shutil.copy2(OLD/"model/linear.npz",support/"parent_linear.npz")
    shutil.copy2(OLD/"model/booster.json",support/"parent_booster.json")
    shutil.copy2(OUT/"encoder.onnx",package/"encoder.onnx")
    shutil.copy2(OUT/"final_head.npz",package/"head.npz")
    for name in ["tokenizer.json","tokenizer_config.json","config.json","README.md","ORIGINAL_MODEL_CARD.md","provenance.json"]:
        shutil.copy2(OLD/"external_model"/name,package/name)
    write_json(package/"model.json",{"schema_version":1,"choice":chosen,"features":828,
        "embedding_dimension":768,"embedding_tensor":"cls_embedding","parent_alpha":1.,
        "known_identity_gate_columns":[14,17,20],"cold_history_fallback":"parent_fifth_round",
        "teacher_permission_user_report":True,"permission_condition":"understand pretrained model",
        "source_encoder_sha256":sha256(OLD/"external_model/onnx/model_quantized.onnx"),
        "encoder_sha256":sha256(package/"encoder.onnx"),"input_runtime":{"batch":16,"max_length":128,"provider":"cpu"},
        "training_rows":len(data["y"]),"previous_kaggle_user_report":.54604,
        "test_parent_max_abs_difference":max_difference,"weights_of_bert_updated":False})
    write_json(OUT/"test_parent_verification.json",{"maximum_difference":max_difference,"matches_previous_054604_submission":True})
    print("第六轮最终训练及打包完成",chosen,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["develop","final"],required=True)
    args=parser.parse_args();develop() if args.stage=="develop" else final()
