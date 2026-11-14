"""最终适配权重的GPU缓存→CPU部署数值核对；不使用课程评分判断通过与否。"""
import json
import numpy as np
import torch
from task2.partial9 import FOLDER,OUT,load_top
from task2.partial9_protocol import tag,run_name
from task2.partial9_cache import cached_batch
from task2.audit import write_json


def main():
    torch.set_num_threads(1)
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    controls=np.load(OUT/"controls.npz");groups=controls["groups"]
    fit=(groups%100==0)&((groups//100)%5!=0)
    length=choice["max_length"];n=128 if length==128 else 32
    if length==256:fit &= np.load(FOLDER/"development/lengths.npy")==128
    positions=np.flatnonzero(fit)[np.linspace(0,int(fit.sum())-1,n,dtype=int)]
    global_positions=positions.copy()
    suffix="" if length==128 else "_256";reference=np.load(FOLDER/f"audit_batch{suffix}.npz")
    folder=FOLDER/"development" if length==128 else FOLDER/"long_development"
    if length==256:positions=np.searchsorted(np.load(FOLDER/"long_positions.npy"),positions)
    top=FOLDER/run_name("final",choice)/f"epoch_{choice['epochs']}.pt"
    cpu=load_top(top,FOLDER/"config").eval();gpu=load_top(top,FOLDER/"config").cuda().eval()
    a=[];b=[];ae=[];be=[]
    with torch.no_grad():
        for first in range(0,n,16):
            original=reference["hidden"][first:first+16];mask=reference["mask"][first:first+16].astype(bool)
            logits,embedding=cpu(torch.from_numpy(original),torch.from_numpy(mask));a.append(logits.numpy());ae.append(embedding.numpy())
            hidden,actual_mask=cached_batch(folder,positions[first:first+16],"cpu")
            padded=torch.zeros((len(hidden),length,768));padded[:,:hidden.shape[1]]=hidden
            attention=torch.zeros((len(hidden),length),dtype=torch.bool);attention[:,:hidden.shape[1]]=actual_mask
            logits,embedding=gpu(padded.cuda(),attention.cuda());b.append(logits.cpu().numpy());be.append(embedding.cpu().numpy())
    a=np.concatenate(a);b=np.concatenate(b)
    def score(values):
        p=np.exp(values-values.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        return p@np.arange(1.,6.)
    if choice.get("objective")=="residual_noise":
        original=np.load(OUT.parent/"round8_20261114/english_development.npz")["logits"][global_positions]
        fixed=controls["final_offset"][global_positions]-original
        difference=np.abs(score(a+fixed)-score(b+fixed))
    else:difference=np.abs(score(a)-score(b))
    proof={"rows":n,"max_length":length,"course_labels_used":False,"logit_mae":float(np.abs(a-b).mean()),
        "rating_mae":float(difference.mean()),"rating_max_error":float(difference.max()),
        "argmax_agreement":float(np.mean(a.argmax(1)==b.argmax(1))),"weights":"actual final adapted state"}
    ae=np.concatenate(ae);be=np.concatenate(be)
    proof["cls_relative_rmse"]=float(np.sqrt(np.mean((ae-be)**2)/np.mean(ae**2)))
    proof["accepted"]=proof["logit_mae"]<.01 and proof["rating_mae"]<.001 and proof["rating_max_error"]<.005 and proof["cls_relative_rmse"]<.002
    write_json(OUT/"adapted_deployment_audit.json",proof);print("实际适配权重部署核验",proof,flush=True)
    if not proof["accepted"]:raise RuntimeError("GPU训练/CPU部署数值核对未通过。")


if __name__=="__main__":main()
