"""第九轮有限内部选型：最后层适配结果与已测第八轮融合。

128个候选：4种目标/学习率组合×2轮次×2长度×(4个凸组合+2个低维头+2个CLS头)。
低维头接入35列结构、旧评分和新分布；另比较加入适配CLS的强正则头。
适配表示在拟合行上并非OOF，是否有收益由内部检查与冻结确认共同判断。
确认组只在配置冻结后使用，不根据消融或Kaggle反馈重新调参。
"""
import argparse
import json
import numpy as np
import pandas as pd
from .audit import ROOT,write_json,sha256
from .partial9_protocol import tag,fit_mask,OUT,FOLDER,objective_suffix,run_name
from .noise_softmax import fit_channel,predict_channel,softmax
from .semantic_fusion import semantic_features
from .round3_train import evaluate


def design(structure,parent,logits,embedding=None):
    blocks=[structure,parent,semantic_features(logits,parent)]
    if embedding is not None:blocks.append(embedding)
    return np.column_stack(blocks).astype(np.float32)


def data():
    controls=dict(np.load(OUT/"controls.npz"))
    raw=np.load(ROOT/"task2/outputs/round3_20261105/development.npz")
    index=np.load(ROOT/"task2/outputs/round6_20261108/selection.npz")["index"]
    np.testing.assert_array_equal(raw["id"][index],controls["id"])
    original=np.load(ROOT/"task2/outputs/round8_20261114/english_development.npz")
    np.testing.assert_array_equal(original["id"],controls["id"]);controls["reference_english_logits"]=original["logits"]
    return controls,raw["x"][index]


def candidate(controls,structure,logits,choice,stage,state=None,embedding=None):
    parent=controls[stage+"_parent"];offset=controls[stage+"_offset"]
    if choice.get("objective")=="residual_noise":logits=offset+logits-controls["reference_english_logits"]
    if choice["route"]=="blend":
        neural=softmax(logits)@np.arange(1.,6.)
        return np.clip(parent+choice["alpha"]*(neural-parent),1,5),None,{}
    if choice["route"]=="cls_head" and embedding is None:raise ValueError("CLS校准缺少更新表示。")
    features=design(structure,parent,logits,embedding if choice["route"]=="cls_head" else None)
    if state is None:
        state,report=fit_channel(features,offset,controls["y"],fit_mask(controls["groups"],stage),
            float(controls["epsilon"]),ridge=choice["ridge"],iterations=250)
    else:report={}
    return np.clip(predict_channel(features,offset,state),1,5),state,report


def metrics(controls,values,mask):
    return evaluate(values[mask],controls["y"][mask],controls["stars"][mask],float(controls["epsilon"]))


def select(screen=False):
    """内部候选可分批计算，但只有128项齐全才允许冻结。

    screen只计算当前已完成的快照；所有结果仍只使用内部检查标签。
    缓存同时绑定计算代码、控制数据和每个实际预测/CLS文件的SHA，
    后续补齐结果时不会把不同版本的快照或指标混用。
    """
    if (OUT/"frozen_selection.json").exists():raise FileExistsError("已经冻结，禁止重复选型。")
    controls,structure=data();active=controls["groups"]%100==0;n=len(active)
    controls={k:v[active] if v.ndim and v.shape[0]==n else v for k,v in controls.items()}
    structure=structure[active];groups=controls["groups"];check=(groups//100)%5==0
    baseline=metrics(controls,controls["inner_parent"],check);records=[]
    fingerprint={name:sha256(path) for name,path in {
        "code":OUT.parents[1]/"partial9_fusion.py",
        "noise":OUT.parents[1]/"noise_softmax.py",
        "semantic":OUT.parents[1]/"semantic_fusion.py",
        "controls":OUT/"controls.npz",
        "protocol":OUT/"selection_protocol.json"}.items()}
    cache_path=OUT/"inner_screen_cache.json"
    cache=json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"fingerprint":fingerprint,"records":{}}
    if cache["fingerprint"]!=fingerprint:raise ValueError("内部计算缓存版本变化，请保留旧记录后重新登记。")
    for run in json.loads((OUT/"selection_protocol.json").read_text(encoding="utf-8"))["objective_runs"]:
        rate=run["learning_rate"];objective=run["objective"]
        for epoch in [1,2]:
            for length in json.loads((OUT/"selection_protocol.json").read_text(encoding="utf-8"))["inference_max_lengths"]:
                suffix=objective_suffix(objective)+("" if length==128 else "_long")
                saved_path=OUT/f"neural_inner_{tag(rate)}_e{epoch}{suffix}.npz"
                if not saved_path.exists():
                    if screen:continue
                    raise FileNotFoundError(saved_path)
                saved=np.load(saved_path);prediction_sha=sha256(saved_path)
                np.testing.assert_array_equal(saved["id"],controls["id"])
                options=[{"route":"blend","alpha":alpha} for alpha in [.25,.5,.75,1.]]
                options += [{"route":"noise_head","ridge":ridge} for ridge in [100.,1000.]]
                options += [{"route":"cls_head","ridge":ridge} for ridge in [1000.,5000.]]
                embedding_path=FOLDER/"cls"/f"neural_inner_{tag(rate)}_e{epoch}{suffix}.npy"
                metadata_path=embedding_path.with_suffix(".json")
                # 文件可能正在写入，只有绑定快照/预测SHA的完成记录存在才算可用。
                ready=metadata_path.exists() and embedding_path.exists()
                embedding=np.load(embedding_path,mmap_mode="r") if ready else None
                embedding_sha=sha256(embedding_path) if ready else None
                if ready:
                    metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
                    checkpoint=FOLDER/run_name("inner",{"learning_rate":rate,"objective":objective})/f"epoch_{epoch}.pt"
                    if metadata["prediction_sha256"]!=prediction_sha or metadata["checkpoint_sha256"]!=sha256(checkpoint):
                        raise ValueError("CLS完成记录与实际预测/快照不一致。")
                    if embedding.shape!=(len(controls["id"]),768):raise ValueError("内部CLS行数或维度不正确。")
                for option in options:
                    if option["route"]=="cls_head" and embedding is None:
                        if screen:continue
                        raise FileNotFoundError(metadata_path)
                    choice={"learning_rate":rate,"epochs":epoch,"max_length":length,"objective":objective,**option}
                    key=json.dumps(choice,sort_keys=True)
                    inputs={"prediction":prediction_sha,"embedding":embedding_sha if option["route"]=="cls_head" else None}
                    previous=cache["records"].get(key)
                    if previous is not None:
                        if previous["inputs"]!=inputs:raise ValueError("候选输入指纹变化，禁止混合已计算的内部指标。")
                        record=previous["record"]
                    else:
                        value,state,report=candidate(controls,structure,saved["logits"],choice,"inner",embedding=embedding)
                        record={**choice,"metrics":metrics(controls,value,check)}
                        cache["records"][key]={"inputs":inputs,"record":record}
                        write_json(cache_path,cache)
                    records.append(record)
                print("内部候选进度",len(records),"/128",objective,tag(rate),epoch,length,flush=True)
    if screen:
        write_json(OUT/"inner_screen_status.json",{"completed":len(records),"total":128,"confirmation_used":False,"selection_frozen":False})
        print("内部预计算完成，尚未冻结",len(records),flush=True);return
    if len(records)!=128:raise ValueError("候选未齐全，禁止冻结。")
    chosen=min(records,key=lambda item:item["metrics"]["noise_model_risk"]["latent_mse_estimate"])
    chosen["internal_improved"]=chosen["metrics"]["noise_model_risk"]["latent_mse_estimate"]<baseline["noise_model_risk"]["latent_mse_estimate"]
    write_json(OUT/"inner_comparison.json",{"baseline":baseline,"candidates":records,"candidate_count":len(records)})
    write_json(OUT/"frozen_selection.json",chosen);print("第九轮冻结选择",chosen,flush=True)


def confirmation():
    controls,structure=data();choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    saved=np.load(OUT/"neural_confirm.npz");np.testing.assert_array_equal(saved["id"],controls["id"])
    embedding=np.load(FOLDER/"cls/neural_confirm.npy",mmap_mode="r") if choice["route"]=="cls_head" else None
    value,state,report=candidate(controls,structure,saved["logits"],choice,"confirm",embedding=embedding)
    if state is not None:np.savez_compressed(OUT/"confirm_fusion.npz",**state)
    write_json(OUT/"confirm_fusion_training.json",report)
    mask=controls["groups"]%100!=0;parent=controls["confirm_parent"]
    old=pd.read_csv(ROOT/"task2/outputs/round8_20261114/confirmation_predictions.csv")
    np.testing.assert_array_equal(old.id,controls["id"][mask]);np.testing.assert_allclose(old.prediction,parent[mask],atol=6e-10,rtol=0)
    unadapted=OUT/"neural_unadapted_long.npz" if choice["max_length"]==256 else ROOT/"task2/outputs/round8_20261114/english_development.npz"
    ablations={"without_adaptation":np.load(unadapted)["logits"]}
    classifier_only=OUT/"neural_confirm_classifier_only.npz"
    if classifier_only.exists():ablations["classifier_only"]=np.load(classifier_only)["logits"]
    values={"baseline":parent,"candidate":value}
    for name,logits in ablations.items():
        ablated_embedding=None
        if choice["route"]=="cls_head":
            if name=="classifier_only":path=FOLDER/"cls/neural_confirm_classifier_only.npy"
            elif choice["max_length"]==256:path=FOLDER/"cls/neural_unadapted_long.npy"
            else:path=ROOT/"task2/outputs/round8_20261114/english_development.embedding.npy"
            ablated_embedding=np.load(path,mmap_mode="r")
        values[name]=candidate(controls,structure,logits,choice,"confirm",embedding=ablated_embedding)[0]
    result={name:metrics(controls,prediction,mask) for name,prediction in values.items()}
    result.update(frozen_before_confirmation=True,historical_validation_reuse=True)
    write_json(OUT/"confirmation.json",result)
    pd.DataFrame({"id":controls["id"][mask],"rating":controls["y"][mask],"groups":controls["groups"][mask],
        **{name if name!="candidate" else "prediction":prediction[mask] for name,prediction in values.items()}}).to_csv(OUT/"confirmation_predictions.csv",index=False)
    print("第九轮冻结确认",result,flush=True)


def fit_final():
    controls,structure=data();choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    proof=json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
    if not choice["internal_improved"] or proof["candidate"]["noise_model_risk"]["latent_mse_estimate"]>=proof["baseline"]["noise_model_risk"]["latent_mse_estimate"]:
        raise RuntimeError("第九轮未共同通过内部/确认，不发布。")
    saved=np.load(OUT/"neural_final.npz");np.testing.assert_array_equal(saved["id"],controls["id"])
    embedding=np.load(FOLDER/"cls/neural_final.npy",mmap_mode="r") if choice["route"]=="cls_head" else None
    _,state,report=candidate(controls,structure,saved["logits"],choice,"final",embedding=embedding)
    if state is not None:np.savez_compressed(OUT/"final_fusion.npz",**state)
    report.update(selection_sha256=sha256(OUT/"frozen_selection.json"),neural_sha256=sha256(OUT/"neural_final.npz"))
    write_json(OUT/"final_fusion_training.json",report)


def final():
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    evidence=json.loads((OUT/"final_fusion_training.json").read_text(encoding="utf-8"))
    if evidence["selection_sha256"]!=sha256(OUT/"frozen_selection.json") or evidence["neural_sha256"]!=sha256(OUT/"neural_final.npz"):
        raise ValueError("最终校准输入变化，拒绝混用参数。")
    state=dict(np.load(OUT/"final_fusion.npz")) if choice["route"]!="blend" else None
    from .round8_fusion import build_test,OUT as OLD
    from .embedding_heads import linear_logits
    x8,_,offset7=build_test({"lexical":40,"english":True})
    offset8=linear_logits(x8,offset7,dict(np.load(OLD/"final_head.npz")));parent=softmax(offset8)@np.arange(1.,6.)
    test=np.load(OUT/"neural_test.npz");raw=np.load(ROOT/"task2/outputs/round3_20261105/test.npz")
    np.testing.assert_array_equal(test["id"],raw["id"])
    logits=test["logits"]
    if choice.get("objective")=="residual_noise":
        logits=offset8+logits-np.load(OLD/"english_test.npz")["logits"]
    if choice["route"]=="blend":values=np.clip(parent+choice["alpha"]*(softmax(logits)@np.arange(1.,6.)-parent),1,5)
    else:
        embedding=np.load(OUT/"neural_test.cls.npy",mmap_mode="r") if choice["route"]=="cls_head" else None
        values=np.clip(predict_channel(design(raw["x"],parent,logits,embedding),offset8,state),1,5)
    pd.DataFrame({"id":test["id"],"rating":values}).to_csv(OUT/"submission.csv",index=False,float_format="%.10f")
    print("第九轮最终预测完成",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["screen","select","confirm","fit-final","final"],required=True)
    args=p.parse_args();{"screen":lambda:select(screen=True),"select":select,"confirm":confirmation,"fit-final":fit_final,"final":final}[args.stage]()
