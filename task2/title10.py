"""第十轮独立标题语义对照：冻结英语网络，只训练自主噪声残差头。

第九轮128候选已经冻结；本轮另登记8个标题候选，不能回头扩展第九轮。
标题向量不使用课程评分。第九轮完整五星概率混合作为新头的固定偏置，
保留已有正文/历史能力；只在内部选型，确认通过后才拟合全部83047行。
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import ROOT,write_json,sha256
from .partial9_protocol import OUT as PARENT,FOLDER as CACHE,fit_mask,tag,objective_suffix
from .partial9_fusion import data as parent_data,design
from .noise_softmax import softmax,fit_channel,predict_channel
from .round3_train import evaluate

OUT=ROOT/"task2/outputs/round10_20261114"
FOLDER=Path.home()/".cache/comp5434-title10"
PROTOCOL={"round":10,"parent":9,"candidate_count":8,"title_encoder_frozen":True,
    "title_max_length":128,"source_revision":"06f020e0dbf909570eb886423bd3af256b855546",
    "heads":[{"route":"noise_head","ridge":100.},{"route":"noise_head","ridge":1000.},
             {"route":"cls_head","ridge":1000.},{"route":"cls_head","ridge":5000.}],
    "alphas":[.5,1.],"selection":"internal_noise_risk","confirmation_used_for_selection":False,
    "historical_validation_reuse":True,"new_kaggle_score":None}


def mixture_offset(old,adapted,alpha):
    """保留两个真实类别分布，不能从连续评分反推五星概率。"""
    q=(1-alpha)*softmax(old)+alpha*softmax(adapted)
    return np.log(np.maximum(q,1e-30))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);FOLDER.mkdir(exist_ok=True)
    path=OUT/"protocol.json"
    if path.exists():raise FileExistsError("本轮已经登记，不覆盖协议。")
    write_json(path,PROTOCOL)
    # 保留原始标题本身，包括字面量None；不把观测评分或预测评分拼入文本。
    for kind in ["development","test"]:
        source=ROOT/f"task2/outputs/round6_20261108/{kind}_text.csv"
        frame=pd.read_csv(source,keep_default_na=False)
        pd.DataFrame({"id":frame.id,"title":"","comment":frame.title.astype(str)}).to_csv(FOLDER/f"{kind}_title.csv",index=False)
    controls=np.load(PARENT/"controls.npz");fit=np.flatnonzero(fit_mask(controls["groups"],"inner"))
    rows=fit[np.linspace(0,len(fit)-1,16,dtype=int)]
    frame=pd.read_csv(FOLDER/"development_title.csv",keep_default_na=False)
    frame.iloc[rows].to_csv(FOLDER/"audit_title.csv",index=False);np.save(FOLDER/"audit_rows.npy",rows)
    write_json(OUT/"input_binding.json",{k:sha256(FOLDER/f"{k}_title.csv") for k in ["development","test"]})


def data(stage):
    controls,structure=parent_data();choice=json.loads((PARENT/"frozen_selection.json").read_text(encoding="utf-8"))
    if choice["route"]!="blend" or choice["objective"]!="residual_noise":raise ValueError("本轮只继承已冻结的残差评分混合。")
    if stage=="inner":
        active=controls["groups"]%100==0;n=len(active)
        controls={k:v[active] if v.ndim and len(v)==n else v for k,v in controls.items()};structure=structure[active]
        suffix=objective_suffix(choice["objective"])+("_long" if choice["max_length"]==256 else "")
        path=PARENT/f"neural_inner_{tag(choice['learning_rate'])}_e{choice['epochs']}{suffix}.npz"
    else:active=np.ones(len(controls["id"]),bool);path=PARENT/f"neural_{stage}.npz"
    raw=np.load(path);np.testing.assert_array_equal(raw["id"],controls["id"])
    effective=controls[stage+"_offset"]+raw["logits"]-controls["reference_english_logits"]
    offset=mixture_offset(controls[stage+"_offset"],effective,choice["alpha"])
    parent=softmax(offset)@np.arange(1.,6.)
    expected=controls[stage+"_parent"]+choice["alpha"]*(softmax(effective)@np.arange(1.,6.)-controls[stage+"_parent"])
    np.testing.assert_allclose(parent,expected,atol=1e-12,rtol=0)
    titles=np.load(FOLDER/"development.npz");np.testing.assert_array_equal(titles["id"][active],controls["id"])
    cls=np.load(FOLDER/"development.embedding.npy",mmap_mode="r")[active]
    return controls,structure,parent,offset,titles["logits"][active],cls


def metric(c,value,mask):return evaluate(value[mask],c["y"][mask],c["stars"][mask],float(c["epsilon"]))


def fit(c,structure,parent,offset,logits,cls,choice,stage):
    features=design(structure,parent,logits,cls if choice["route"]=="cls_head" else None)
    state,report=fit_channel(features,offset,c["y"],fit_mask(c["groups"],stage),float(c["epsilon"]),ridge=choice["ridge"],iterations=250)
    changed=predict_channel(features,offset,state)
    value=np.clip(parent+choice["alpha"]*(changed-parent),1,5)
    return value,state,report


def select():
    if (OUT/"frozen_selection.json").exists():raise FileExistsError("本轮已经冻结。")
    if not json.loads((OUT/"gpu_fidelity.json").read_text(encoding="utf-8"))["accepted"]:raise ValueError("标题GPU数值核验未通过。")
    c,x,parent,offset,logits,cls=data("inner");check=(c["groups"]//100)%5==0
    baseline=metric(c,parent,check);records=[]
    previous=json.loads((PARENT/"frozen_selection.json").read_text(encoding="utf-8"))["metrics"]
    np.testing.assert_allclose(baseline["noise_model_risk"]["latent_mse_estimate"],previous["noise_model_risk"]["latent_mse_estimate"],atol=1e-12,rtol=0)
    for head in PROTOCOL["heads"]:
        value,_,_=fit(c,x,parent,offset,logits,cls,{**head,"alpha":1.},"inner")
        for alpha in PROTOCOL["alphas"]:
            p=np.clip(parent+alpha*(value-parent),1,5)
            records.append({**head,"alpha":alpha,"metrics":metric(c,p,check)})
    chosen=min(records,key=lambda row:row["metrics"]["noise_model_risk"]["latent_mse_estimate"])
    chosen["internal_improved"]=chosen["metrics"]["noise_model_risk"]["latent_mse_estimate"]<baseline["noise_model_risk"]["latent_mse_estimate"]
    write_json(OUT/"inner_comparison.json",{"baseline":baseline,"candidates":records,"candidate_count":len(records)})
    write_json(OUT/"frozen_selection.json",chosen);print("第十轮标题冻结选择",chosen,flush=True)


def confirm():
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    if not choice["internal_improved"]:raise RuntimeError("标题分支内部未胜出，保持第九轮。")
    c,x,parent,offset,logits,cls=data("confirm");mask=c["groups"]%100!=0
    # 第九轮相同确认预测必须完全复现，防止混用final偏置或快照。
    previous=pd.read_csv(PARENT/"confirmation_predictions.csv")
    np.testing.assert_array_equal(previous.id,c["id"][mask]);np.testing.assert_allclose(previous.prediction,parent[mask],atol=6e-10,rtol=0)
    value,state,report=fit(c,x,parent,offset,logits,cls,choice,"confirm")
    proof={"baseline":metric(c,parent,mask),"candidate":metric(c,value,mask),"frozen_before_confirmation":True,"historical_validation_reuse":True}
    write_json(OUT/"confirmation.json",proof);write_json(OUT/"confirm_training.json",report)
    pd.DataFrame({"id":c["id"][mask],"rating":c["y"][mask],"groups":c["groups"][mask],
        "baseline":parent[mask],"prediction":value[mask]}).to_csv(OUT/"confirmation_predictions.csv",index=False)
    print("第十轮标题冻结确认",proof,flush=True)


def final_fit():
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"));proof=json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
    if not choice["internal_improved"] or proof["candidate"]["noise_model_risk"]["latent_mse_estimate"]>=proof["baseline"]["noise_model_risk"]["latent_mse_estimate"]:
        raise RuntimeError("标题分支未同时通过内部/确认，禁止发布。")
    c,x,parent,offset,logits,cls=data("final")
    values,state,report=fit(c,x,parent,offset,logits,cls,choice,"final")
    np.savez_compressed(OUT/"final_head.npz",**state);write_json(OUT/"final_training.json",report)
    # 用实际最终头检查GPU训练表示与CPU部署表示的评分差，防止标准化放大微小表示误差。
    # 审计输入仅来自原内部拟合行；接受门槛比较数值，不使用其评分标签。
    rows=np.load(FOLDER/"audit_rows.npy");reference=np.load(FOLDER/"audit.npz")
    np.testing.assert_array_equal(reference["id"],c["id"][rows])
    cpu_cls=np.load(FOLDER/"audit.embedding.npy") if choice["route"]=="cls_head" else None
    cpu_features=design(x[rows],parent[rows],reference["logits"],cpu_cls)
    cpu_prediction=np.clip(parent[rows]+choice["alpha"]*(predict_channel(cpu_features,offset[rows],state)-parent[rows]),1,5)
    error=np.abs(cpu_prediction-values[rows])
    audit={"rows":len(rows),"rating_mae":float(error.mean()),"rating_max_error":float(error.max()),
        "labels_used_for_acceptance":False,"weights":"actual final title residual head",
        "accepted":bool(error.mean()<.0001 and error.max()<.001)}
    write_json(OUT/"final_title_deployment_audit.json",audit);print("标题最终头部署核验",audit,flush=True)
    if not audit["accepted"]:raise RuntimeError("标题最终评分头的GPU/CPU表示差异过大，拒绝发布。")


def export():
    """从训练产物路径生成测试预测，随后用独立原始CSV路径验证。"""
    from .round8_fusion import build_test,OUT as EIGHT
    from .embedding_heads import linear_logits
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    parent_choice=json.loads((PARENT/"frozen_selection.json").read_text(encoding="utf-8"))
    x8,_,offset7=build_test({"lexical":40,"english":True})
    offset8=linear_logits(x8,offset7,dict(np.load(EIGHT/"final_head.npz")))
    raw=np.load(PARENT/"neural_test.npz");reference=np.load(EIGHT/"english_test.npz")
    np.testing.assert_array_equal(raw["id"],reference["id"])
    offset=mixture_offset(offset8,offset8+raw["logits"]-reference["logits"],parent_choice["alpha"])
    parent=softmax(offset)@np.arange(1.,6.)
    old=pd.read_csv(PARENT/"submission.csv");np.testing.assert_array_equal(old.id,raw["id"])
    np.testing.assert_allclose(parent,old.rating,atol=6e-10,rtol=0)
    title=np.load(FOLDER/"test.npz");np.testing.assert_array_equal(title["id"],raw["id"])
    structure=np.load(ROOT/"task2/outputs/round3_20261105/test.npz")["x"]
    cls=np.load(FOLDER/"test.embedding.npy",mmap_mode="r") if choice["route"]=="cls_head" else None
    features=design(structure,parent,title["logits"],cls)
    changed=predict_channel(features,offset,dict(np.load(OUT/"final_head.npz")))
    value=np.clip(parent+choice["alpha"]*(changed-parent),1,5)
    pd.DataFrame({"id":raw["id"],"rating":value}).to_csv(OUT/"submission.csv",index=False,float_format="%.10f")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["prepare","select","confirm","fit-final","export"],required=True)
    args=p.parse_args();{"prepare":prepare,"select":select,"confirm":confirm,"fit-final":final_fit,"export":export}[args.stage]()
