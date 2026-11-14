"""第七轮五星词分类与第六轮BERT头融合；内部分组选型后只确认一次。

五星词模型在全部原外层验证行之外训练。组合器也严格分inner/confirm/final，
确认比较使用confirm旧头，不能读取见过确认标签的final预测充当基线。
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from .audit import ROOT,write_json
from .features import load_partition,TEXT_DIM
from .history_prepare import OUT
from .history_train import masks
from .embedding_features import development,stage_features,OUT as OLD
from .embedding_heads import linear_logits
from .noise_softmax import softmax,fit_channel,predict_channel
from .semantic_fusion import semantic_features
from .round3_train import evaluate


def old_logits(features,offset,head):
    """按旧MLP相同float32路径恢复其五类logits，便于直接做噪声通道残差。"""
    anchor={k:head["anchor_"+k] for k in ["center","scale","weights"]}
    output=np.empty((len(features),5),dtype=np.float32)
    for first in range(0,len(features),2048):
        batch=features[first:first+2048]
        z=((batch-anchor["center"])/anchor["scale"]).astype(np.float32)
        logits=linear_logits(batch,offset[first:first+len(batch)],anchor).astype(np.float32)
        logits+=np.tanh(z@head["w1"]+head["b1"])@head["w2"]+head["b2"]
        output[first:first+len(batch)]=logits
    return output


def prepare(iteration):
    """只提取固定原验证行和测试，不用文本评分来挑选容易的验证记录。"""
    folder=OUT/"sparse_scaled_split";model=folder/(f"checkpoint_{iteration}.npz" if iteration!=20 else "weights.npz")
    state=np.load(model);w,bias=state["weights"],state["bias"]
    parts=json.loads((ROOT/"task2/cache/manifest.json").read_text(encoding="utf-8"))["parts"]
    ids=[];logits=[]
    for part in parts:
        x,y,fit,valid,identifier=load_partition(part["prefix"],"split","hybrid")
        ids.append(identifier[valid]);logits.append(x[valid]@w+bias)
    identifiers=np.concatenate(ids);logits=np.concatenate(logits)
    expected=np.load(ROOT/"task2/outputs/round3_20261105/development.npz")["id"]
    np.testing.assert_array_equal(identifiers,expected)
    text=sparse.load_npz(ROOT/"task2/cache/test.text.npz");numeric=np.load(ROOT/"task2/cache/test.full.numeric.npy")
    test_logits=text@w[:TEXT_DIM]+numeric@w[TEXT_DIM:]+bias
    np.savez_compressed(OUT/f"sparse_predictions_{iteration}.npz",id=identifiers,logits=logits,test_logits=test_logits)
    print("五星词分布提取",iteration,len(identifiers),flush=True)


def stage_data(stage,iteration):
    data,index,logits,embedding=development()
    x,parent5=stage_features(index,logits,embedding,stage)
    name={"inner":"inner_mlp_12.npz","confirm":"confirm_head.npz","final":"final_head.npz"}[stage]
    offset=old_logits(x,logits,dict(np.load(OLD/name)))
    # 旧头的softmax一直以float32计算；按原精度取期望以保证旧CSV可复现。
    parent6=softmax(offset)@np.arange(1.,6.)
    scores=np.load(OUT/f"sparse_predictions_{iteration}.npz")
    matrix=np.column_stack([x[:,:60],parent6,semantic_features(scores["logits"][index],parent6)]).astype(np.float32)
    return data,matrix,parent6,offset


def develop():
    if (OUT/"sparse_frozen.json").exists():raise FileExistsError("五星融合已冻结。")
    records=[];began=time.perf_counter();baseline=None
    for iteration in [10,20]:
        data,x,parent,offset=stage_data("inner",iteration);fits,check,confirm=masks(data)
        baseline=evaluate(parent[check],data["y"][check],data["stars"][check],float(data["epsilon"]))
        for ridge in [100.,1000.]:
            state,report=fit_channel(x,offset,data["y"],fits["inner"],float(data["epsilon"]),ridge=ridge,iterations=200)
            np.savez_compressed(OUT/f"sparse_inner_{iteration}_{int(ridge)}.npz",**state)
            changed=predict_channel(x,offset,state)
            for alpha in [.5,1.]:
                p=np.clip(parent+alpha*(changed-parent),1,5)
                records.append({"iteration":iteration,"ridge":ridge,"alpha":alpha,
                    "metrics":evaluate(p[check],data["y"][check],data["stars"][check],float(data["epsilon"]))})
            print("五星词融合内部模型",iteration,ridge,flush=True)
    chosen=min(records,key=lambda row:row["metrics"]["noise_model_risk"]["latent_mse_estimate"])
    write_json(OUT/"sparse_inner_comparison.json",{"baseline":baseline,"candidates":records,"seconds":time.perf_counter()-began})
    chosen["internal_improved"]=chosen["metrics"]["noise_model_risk"]["latent_mse_estimate"]<baseline["noise_model_risk"]["latent_mse_estimate"]
    write_json(OUT/"sparse_frozen.json",chosen);print("五星融合冻结",chosen,flush=True)


def finish(stage):
    choice=json.loads((OUT/"sparse_frozen.json").read_text(encoding="utf-8"))
    data,x,parent,offset=stage_data(stage,choice["iteration"]);fits,check,confirm=masks(data)
    state,report=fit_channel(x,offset,data["y"],fits[stage],float(data["epsilon"]),ridge=choice["ridge"],iterations=200)
    np.savez_compressed(OUT/f"sparse_{stage}_head.npz",**state);write_json(OUT/f"sparse_{stage}_training.json",report)
    changed=predict_channel(x,offset,state);p=np.clip(parent+choice["alpha"]*(changed-parent),1,5)
    if stage=="confirm":
        ablated,_=fit_channel(x[:,:61],offset,data["y"],fits[stage],float(data["epsilon"]),ridge=choice["ridge"],iterations=200)
        no_new=predict_channel(x[:,:61],offset,ablated);no_new=np.clip(parent+choice["alpha"]*(no_new-parent),1,5)
        result={name:evaluate(value[confirm],data["y"][confirm],data["stars"][confirm],float(data["epsilon"]))
                for name,value in [("baseline",parent),("candidate",p),("without_sparse",no_new)]}
        write_json(OUT/"sparse_confirmation.json",result)
        pd.DataFrame({"id":data["id"][confirm],"rating":data["y"][confirm],"baseline":parent[confirm],
            "prediction":p[confirm],"without_sparse":no_new[confirm],"groups":data["groups"][confirm]}).to_csv(OUT/"sparse_confirmation_predictions.csv",index=False)
        print("五星词融合确认",result,flush=True)
    else:
        result=json.loads((OUT/"sparse_confirmation.json").read_text(encoding="utf-8"))
        if not choice["internal_improved"] or result["candidate"]["noise_model_risk"]["latent_mse_estimate"]>=result["baseline"]["noise_model_risk"]["latent_mse_estimate"]:
            raise RuntimeError("五星词融合未通过确认，不发布。")
        logits=np.load(OLD/"test.npz")["logits"];embedding=np.load(OLD/"test.embedding.npy",mmap_mode="r")
        features,parent5=stage_features(None,logits,embedding,"final")
        offset=old_logits(features,logits,dict(np.load(OLD/"final_head.npz")))
        parent=softmax(offset)@np.arange(1.,6.)
        np.testing.assert_allclose(parent,pd.read_csv(OLD/"submission.csv").rating,atol=6e-10,rtol=0)
        new_logits=np.load(OUT/f"sparse_predictions_{choice['iteration']}.npz")["test_logits"]
        features=np.column_stack([features[:,:60],parent,semantic_features(new_logits,parent)]).astype(np.float32)
        p=np.clip(parent+choice["alpha"]*(predict_channel(features,offset,state)-parent),1,5)
        pd.DataFrame({"id":np.load(OLD/"test.npz")["id"],"rating":p}).to_csv(OUT/"sparse_submission.csv",index=False,float_format="%.10f")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","develop","confirm","final"],required=True)
    args=parser.parse_args()
    if args.stage=="prepare":
        for iteration in [10,20]:prepare(iteration)
    elif args.stage=="develop":develop()
    else:finish(args.stage)
