"""第八轮特征和噪声融合：第七轮为基线，新增TF-IDF/英语CLS后再冻结确认。

共享旧模型相应阶段的logits，严格分inner/confirm/final；不能把final旧头
当作确认基线。模型选择仅用原内部检查行，确认标签不得反向影响参数。
"""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import sparse
from .audit import ROOT,write_json
from .sparse_fusion import stage_data as stage7
from .history_prepare import OUT as PREVIOUS
from .history_train import masks
from .embedding_features import development,stage_features,OUT as EMBEDDING
from .sparse_fusion import old_logits
from .embedding_heads import linear_logits
from .noise_softmax import softmax,fit_channel,predict_channel
from .semantic_fusion import semantic_features
from .tfidf_sparse import partition,transform
from .features import TEXT_DIM
from .round3_train import evaluate
from . import round8_joint as joint
from .embedding_heads import fit_mlp

OUT=ROOT/"task2/outputs/round8_20261114"


def base(stage):
    target=OUT/f"base_{stage}.npz"
    if target.exists():return dict(np.load(target))
    data,x,parent6,offset6=stage7(stage,20)
    name={"inner":"sparse_inner_20_100.npz","confirm":"sparse_confirm_head.npz","final":"sparse_final_head.npz"}[stage]
    offset7=linear_logits(x,offset6,dict(np.load(PREVIOUS/name)))
    parent7=softmax(offset7)@np.arange(1.,6.)
    result={"id":data["id"],"x":np.column_stack([x,parent7]).astype(np.float32),"parent":parent7,"offset":offset7}
    np.savez_compressed(target,**result)
    if stage=="confirm":
        _,_,confirm=masks(data);previous=pd.read_csv(PREVIOUS/"sparse_confirmation_predictions.csv")
        np.testing.assert_array_equal(previous.id,data["id"][confirm])
        np.testing.assert_allclose(previous.prediction,parent7[confirm],atol=6e-10,rtol=0)
    return result


def lexical(iteration):
    """新IDF分类器的全部原验证/测试logits，只有输入变换使用无标签IDF参数。"""
    model=OUT/"tfidf_split"/(f"checkpoint_{iteration}.npz" if iteration!=40 else "weights.npz")
    state=np.load(model);weights,bias=state["weights"],state["bias"]
    config=json.loads((OUT/"tfidf_split/training.json").read_text(encoding="utf-8"))["representation"]
    manifest=json.loads((ROOT/"task2/cache/manifest.json").read_text(encoding="utf-8"))
    ids=[];logits=[]
    for part in manifest["parts"]:
        x,y,fit,valid,identifier=partition(part["prefix"],"split",config)
        ids.append(identifier[valid]);logits.append(x[valid]@weights+bias)
    ids=np.concatenate(ids);logits=np.concatenate(logits)
    np.testing.assert_array_equal(ids,np.load(ROOT/"task2/outputs/round3_20261105/development.npz")["id"])
    text=sparse.load_npz(ROOT/"task2/cache/test.text.npz")
    idf=np.load(Path(config["folder"])/"idf.npy");text=transform(text,idf)
    numeric=np.load(ROOT/"task2/cache/test.full.numeric.npy")
    test_logits=text@weights[:TEXT_DIM]+numeric@weights[TEXT_DIM:]+bias
    np.savez_compressed(OUT/f"lexical_{iteration}.npz",id=ids,logits=logits,test_logits=test_logits)
    print("第八轮词分布完成",iteration,flush=True)


def matrix(stage,choice):
    data,index,*_=development();old=base(stage);blocks=[old["x"]]
    if choice["lexical"]:
        scores=np.load(OUT/f"lexical_{choice['lexical']}.npz")
        np.testing.assert_array_equal(scores["id"][index],data["id"])
        blocks.append(semantic_features(scores["logits"][index],old["parent"]))
    if choice["english"]:
        scores=np.load(OUT/"english_development.npz")
        np.testing.assert_array_equal(scores["id"],data["id"])
        blocks.append(semantic_features(scores["logits"],old["parent"]))
        blocks.append(np.load(OUT/"english_development.embedding.npy",mmap_mode="r"))
    return data,np.column_stack(blocks).astype(np.float32),old["parent"],old["offset"]


def joint_matrix():
    """原35列和Huber预测均在整个原验证集之外拟合，两个编码器完全冻结。"""
    data,index,bert_logits,bert_cls=development()
    original=np.load(ROOT/"task2/outputs/round3_20261105/development.npz")
    auxiliary=np.load(ROOT/"task2/outputs/round4_20261105/auxiliary_split.npz")["prediction"][index].astype(np.float32)
    lexical=np.load(OUT/"lexical_40.npz")["logits"][index]
    english=np.load(OUT/"english_development.npz")
    np.testing.assert_array_equal(english["id"],data["id"])
    english_cls=np.load(OUT/"english_development.embedding.npy",mmap_mode="r")
    x,offset,layout=joint.assemble(original["x"][index],auxiliary,bert_logits,bert_cls,lexical,english["logits"],english_cls)
    write_json(OUT/"joint_layout.json",layout)
    return data,x,offset,layout,bert_logits,english["logits"]


def develop(include_english):
    """只在内部检查组排名。冻结文件存在就拒绝再次选择，确认分数不参与排名。"""
    if (OUT/"frozen_selection.json").exists():raise FileExistsError("第八轮配置已经冻结。")
    pool=[{"lexical":i,"english":False,"ridges":[100.,1000.]} for i in [20,40]]
    if include_english:
        pool += [{"lexical":0,"english":True,"ridges":[1000.,5000.]}]
        pool += [{"lexical":i,"english":True,"ridges":[1000.]} for i in [20,40]]
    write_json(OUT/"selection_protocol.json",{"pool":pool,"alphas":[.5,1.],"choice_before_confirmation":True,
        "joint_linear_ridges":[1000.,5000.] if include_english else [],"joint_mlp_epochs":[6,12] if include_english else [],
        "known_best_kaggle_user_report":.53719,"historical_validation_reuse":True})
    records=[];began=time.perf_counter();baseline=None
    for candidate in pool:
        data,x,parent,offset=matrix("inner",candidate);fits,check,confirm=masks(data)
        baseline=evaluate(parent[check],data["y"][check],data["stars"][check],float(data["epsilon"]))
        for ridge in candidate["ridges"]:
            name=f"inner_l{candidate['lexical']}_e{int(candidate['english'])}_r{int(ridge)}"
            if (OUT/(name+".npz")).exists():
                state=dict(np.load(OUT/(name+".npz")));report=json.loads((OUT/(name+".json")).read_text(encoding="utf-8"))
            else:
                state,report=fit_channel(x,offset,data["y"],fits["inner"],float(data["epsilon"]),ridge=ridge,iterations=250)
                np.savez_compressed(OUT/(name+".npz"),**state);write_json(OUT/(name+".json"),report)
            changed=predict_channel(x,offset,state)
            for alpha in [.5,1.]:
                p=np.clip(parent+alpha*(changed-parent),1,5)
                records.append({"family":"residual","lexical":candidate["lexical"],"english":candidate["english"],"ridge":ridge,"alpha":alpha,
                    "metrics":evaluate(p[check],data["y"][check],data["stars"][check],float(data["epsilon"]))})
            print("第八轮内部模型",name,"loss",report["objective"],flush=True)
            write_json(OUT/"inner_comparison.json",{"baseline":baseline,"candidates":records,"seconds":time.perf_counter()-began})
        del x;gc.collect()
    if include_english:
        data,x,offset,layout,_,_=joint_matrix();fits,check,confirm=masks(data);parent=base("inner")["parent"];anchor=None
        for ridge in [1000.,5000.]:
            choice={"family":"joint_linear","ridge":ridge,"lexical":40,"english":True}
            state,report=joint.fit(x,offset,data["y"],fits["inner"],float(data["epsilon"]),choice)
            np.savez_compressed(OUT/f"inner_joint_linear_{int(ridge)}.npz",**state)
            write_json(OUT/f"inner_joint_linear_{int(ridge)}.json",report)
            changed=joint.clean_prediction(x,offset,state,choice)
            for alpha in [.5,1.]:
                p=np.clip(parent+alpha*(changed-parent),1,5)
                records.append({**choice,"alpha":alpha,"metrics":evaluate(p[check],data["y"][check],data["stars"][check],float(data["epsilon"]))})
            if ridge==1000.:anchor=state
            print("联合线性头内部训练",ridge,report["objective"],flush=True)
        states,report=fit_mlp(x,offset,data["y"],fits["inner"],float(data["epsilon"]),anchor,epochs=12)
        write_json(OUT/"inner_joint_mlp.json",report)
        for epochs in [6,12]:
            state=states[epochs];choice={"family":"joint_mlp","ridge":1000.,"epochs":epochs,"lexical":40,"english":True}
            np.savez_compressed(OUT/f"inner_joint_mlp_{epochs}.npz",**state)
            changed=joint.clean_prediction(x,offset,state,choice)
            for alpha in [.5,1.]:
                p=np.clip(parent+alpha*(changed-parent),1,5)
                records.append({**choice,"alpha":alpha,"metrics":evaluate(p[check],data["y"][check],data["stars"][check],float(data["epsilon"]))})
        write_json(OUT/"inner_comparison.json",{"baseline":baseline,"candidates":records,"seconds":time.perf_counter()-began})
    # 比较的是同一噪声假设下的MSE；sqrt单调，不因有限样本负估计而擅自裁剪后排名。
    # 这里没有加入确认组指标，也不根据线上分数重新调整候选融合系数。
    chosen=min(records,key=lambda row:row["metrics"]["noise_model_risk"]["latent_mse_estimate"])
    chosen["internal_improved"]=chosen["metrics"]["noise_model_risk"]["latent_mse_estimate"]<baseline["noise_model_risk"]["latent_mse_estimate"]
    write_json(OUT/"frozen_selection.json",chosen);print("第八轮冻结配置",chosen,flush=True)


def prefit():
    """英语编码时先拟合独立的词分布候选，尚不选型、不读取确认评分。"""
    for iteration in [20,40]:
        data,x,parent,offset=matrix("inner",{"lexical":iteration,"english":False})
        fits,_,_=masks(data)
        for ridge in [100.,1000.]:
            name=f"inner_l{iteration}_e0_r{int(ridge)}"
            if (OUT/(name+".npz")).exists():continue
            state,report=fit_channel(x,offset,data["y"],fits["inner"],float(data["epsilon"]),ridge=ridge,iterations=250)
            np.savez_compressed(OUT/(name+".npz"),**state);write_json(OUT/(name+".json"),report)
            print("词分布候选预拟合完成",name,flush=True)


def fit_stage(stage):
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    if choice.get("family", "residual").startswith("joint_"):
        return fit_joint_stage(stage,choice)
    data,x,parent,offset=matrix(stage,choice);fits,check,confirm=masks(data)
    state,report=fit_channel(x,offset,data["y"],fits[stage],float(data["epsilon"]),ridge=choice["ridge"],iterations=250)
    np.savez_compressed(OUT/f"{stage}_head.npz",**state);write_json(OUT/f"{stage}_training.json",report)
    changed=predict_channel(x,offset,state);p=np.clip(parent+choice["alpha"]*(changed-parent),1,5)
    if stage=="confirm":
        # 同一个头移除全部新增输入；只做归因，不允许据确认消融结果再选模型。
        ablated,_=fit_channel(x[:,:81],offset,data["y"],fits[stage],float(data["epsilon"]),ridge=choice["ridge"],iterations=250)
        no_new=np.clip(parent+choice["alpha"]*(predict_channel(x[:,:81],offset,ablated)-parent),1,5)
        result={name:evaluate(values[confirm],data["y"][confirm],data["stars"][confirm],float(data["epsilon"]))
            for name,values in [("baseline",parent),("candidate",p),("without_new_features",no_new)]}
        result.update(frozen_before_confirmation=True,historical_validation_reuse=True)
        write_json(OUT/"confirmation.json",result)
        pd.DataFrame({"id":data["id"][confirm],"rating":data["y"][confirm],"baseline":parent[confirm],"prediction":p[confirm],
            "without_new_features":no_new[confirm],"groups":data["groups"][confirm]}).to_csv(OUT/"confirmation_predictions.csv",index=False)
        print("第八轮冻结确认",result,flush=True)
    else:
        result=json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
        if not choice["internal_improved"] or result["candidate"]["noise_model_risk"]["latent_mse_estimate"]>=result["baseline"]["noise_model_risk"]["latent_mse_estimate"]:
            raise RuntimeError("第八轮未通过确认，不发布。")
        test_matrix,parent,offset=build_test(choice)
        p=np.clip(parent+choice["alpha"]*(predict_channel(test_matrix,offset,state)-parent),1,5)
        pd.DataFrame({"id":np.load(EMBEDDING/"test.npz")["id"],"rating":p}).to_csv(OUT/"submission.csv",index=False,float_format="%.10f")


def fit_joint_stage(stage,choice):
    """冻结配置后重训：confirm只读group0标签，final才允许使用全部开发标签。"""
    data,x,offset,layout,bert_logits,english_logits=joint_matrix()
    fits,check,confirm=masks(data);parent=base(stage)["parent"]
    state,report=joint.fit(x,offset,data["y"],fits[stage],float(data["epsilon"]),choice)
    np.savez_compressed(OUT/f"{stage}_head.npz",**state);write_json(OUT/f"{stage}_training.json",report)
    changed=joint.clean_prediction(x,offset,state,choice);p=np.clip(parent+choice["alpha"]*(changed-parent),1,5)
    if stage=="confirm":
        values={"baseline":parent,"candidate":p}
        ablations={"without_new_features":({"lexical","english_distribution","english_cls"},True),
            "without_english":({"english_distribution","english_cls"},True),
            "without_lexical":({"lexical"},False),"without_english_cls":({"english_cls"},False)}
        # 消融必须连同对应冻结先验一起移除，不能从特征列删掉英语却在offset里保留。
        # 消融只用于解释候选，不允许看到确认结果后替换已冻结配置。
        for name,(remove,remove_offset) in ablations.items():
            columns=joint.keep_columns(layout,remove)
            ablated_offset=joint.frozen_prior(bert_logits) if remove_offset else offset
            model,timing=joint.fit(x[:,columns],ablated_offset,data["y"],fits[stage],float(data["epsilon"]),choice)
            value=joint.clean_prediction(x[:,columns],ablated_offset,model,choice)
            values[name]=np.clip(parent+choice["alpha"]*(value-parent),1,5)
            write_json(OUT/(name+"_training.json"),timing)
            print("联合头固定消融",name,flush=True)
        result={name:evaluate(value[confirm],data["y"][confirm],data["stars"][confirm],float(data["epsilon"])) for name,value in values.items()}
        result.update(frozen_before_confirmation=True,historical_validation_reuse=True,joint_inputs_fit_outside_outer_validation=True)
        write_json(OUT/"confirmation.json",result)
        pd.DataFrame({"id":data["id"][confirm],"rating":data["y"][confirm],"baseline":parent[confirm],"prediction":p[confirm],
            "groups":data["groups"][confirm],**{name:value[confirm] for name,value in values.items() if name.startswith("without_")}}).to_csv(OUT/"confirmation_predictions.csv",index=False)
        print("联合头冻结确认",result,flush=True)
    else:
        result=json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
        if not choice["internal_improved"] or result["candidate"]["noise_model_risk"]["latent_mse_estimate"]>=result["baseline"]["noise_model_risk"]["latent_mse_estimate"]:
            raise RuntimeError("联合头未通过确认，不发布。")
        tx,parent,toffset=joint_test()
        prediction=joint.clean_prediction(tx,toffset,state,choice)
        p=np.clip(parent+choice["alpha"]*(prediction-parent),1,5)
        pd.DataFrame({"id":np.load(EMBEDDING/"test.npz")["id"],"rating":p}).to_csv(OUT/"submission.csv",index=False,float_format="%.10f")


def joint_test():
    original=np.load(ROOT/"task2/outputs/round3_20261105/test.npz")
    auxiliary=np.load(ROOT/"task2/outputs/round4_20261105/auxiliary_full.npz")["prediction"].astype(np.float32)
    bert=np.load(EMBEDDING/"test.npz");bert_cls=np.load(EMBEDDING/"test.embedding.npy",mmap_mode="r")
    lexical=np.load(OUT/"lexical_40.npz")["test_logits"]
    english=np.load(OUT/"english_test.npz");english_cls=np.load(OUT/"english_test.embedding.npy",mmap_mode="r")
    np.testing.assert_array_equal(original["id"],english["id"])
    x,offset,_=joint.assemble(original["x"],auxiliary,bert["logits"],bert_cls,lexical,english["logits"],english_cls)
    _,parent,_=build_test({"lexical":0,"english":False})
    return x,parent,offset


def build_test(choice):
    logits=np.load(EMBEDDING/"test.npz")["logits"];embedding=np.load(EMBEDDING/"test.embedding.npy",mmap_mode="r")
    x6,parent5=stage_features(None,logits,embedding,"final")
    offset6=old_logits(x6,logits,dict(np.load(EMBEDDING/"final_head.npz")));parent6=softmax(offset6)@np.arange(1.,6.)
    original=np.load(PREVIOUS/"sparse_predictions_20.npz")["test_logits"]
    x7=np.column_stack([x6[:,:60],parent6,semantic_features(original,parent6)]).astype(np.float32)
    offset7=linear_logits(x7,offset6,dict(np.load(PREVIOUS/"sparse_final_head.npz")))
    parent7=softmax(offset7)@np.arange(1.,6.)
    np.testing.assert_allclose(parent7,pd.read_csv(PREVIOUS/"sparse_submission.csv").rating,atol=6e-10,rtol=0)
    blocks=[np.column_stack([x7,parent7])]
    if choice["lexical"]:blocks.append(semantic_features(np.load(OUT/f"lexical_{choice['lexical']}.npz")["test_logits"],parent7))
    if choice["english"]:
        scores=np.load(OUT/"english_test.npz");np.testing.assert_array_equal(scores["id"],np.load(EMBEDDING/"test.npz")["id"])
        blocks.extend([semantic_features(scores["logits"],parent7),np.load(OUT/"english_test.embedding.npy",mmap_mode="r")])
    return np.column_stack(blocks).astype(np.float32),parent7,offset7


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["prepare","prefit","develop","confirm","final"],required=True)
    p.add_argument("--english",action="store_true");args=p.parse_args()
    if args.stage=="prepare":
        for iteration in [20,40]:lexical(iteration)
    elif args.stage=="develop":develop(args.english)
    elif args.stage=="prefit":prefit()
    else:fit_stage(args.stage)
