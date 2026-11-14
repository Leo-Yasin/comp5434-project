"""固定编码器的确认消融：直接用已核验冻结CLS训练原pooler/classifier。

编码器始终eval，既不更新权重也不反复计算同一个表示；与正式适配模型使用
同一允许标签行、目标、有效批量和学习率轨迹。仅用于固定确认归因，不参与选型。
"""
import json
import math
import time
import numpy as np
import torch
from .audit import ROOT,write_json
from .partial9 import load_top
from .partial9_protocol import OUT,FOLDER,fit_mask,run_name,teacher_posterior
from .finetune_protocol import forward_noise_loss
from .partial9_cls import store_cls


def train_control(stage,rate,epochs,max_length,objective):
    torch.set_num_threads(1);torch.manual_seed(5434);torch.cuda.manual_seed_all(5434)
    torch.cuda.set_per_process_memory_fraction(.8);torch.use_deterministic_algorithms(True)
    controls=np.load(OUT/"controls.npz");rows=np.flatnonzero(fit_mask(controls["groups"],stage))
    labels=controls["y"].astype(np.int64)-1;eps=float(controls["epsilon"])
    train_source=ROOT/"task2/outputs/round8_20261114/english_development.embedding.npy"
    source=FOLDER/"cls/neural_unadapted_long.npy" if max_length==256 else train_source
    embedding=np.load(source,mmap_mode="r")
    # 正式编码层始终在128token训练，消融也必须使用128的冻结CLS训练。
    # 256仅用于推理对照，不能因选中长推理就偷偷改变控制组的训练上下文。
    vectors=torch.from_numpy(np.array(np.load(train_source,mmap_mode="r"))).cuda()
    model=load_top().cuda();model.block.requires_grad_(False);model.train();model.block.eval()
    initial={k:v.detach().cpu().clone() for k,v in model.block.state_dict().items()}
    optimizer=torch.optim.AdamW(list(model.pooler.parameters())+list(model.classifier.parameters()),lr=rate*5,weight_decay=.01,foreach=False)
    choice={"learning_rate":rate,"objective":objective};destination=FOLDER/run_name(stage,choice,True);destination.mkdir(exist_ok=False)
    teacher=None
    if objective=="teacher_posterior":
        teacher=np.zeros_like(controls[stage+"_offset"]);raw=controls[stage+"_offset"][rows]
        probability=np.exp(raw-raw.max(1,keepdims=True));probability/=probability.sum(1,keepdims=True);teacher[rows]=probability
    fixed_offset=None
    if objective=="residual_noise":
        original=np.load(ROOT/"task2/outputs/round8_20261114/english_development.npz")
        np.testing.assert_array_equal(original["id"],controls["id"])
        fixed_offset=(controls[stage+"_offset"]-original["logits"]).astype(np.float32)
    rng=np.random.default_rng(5434);total=2*math.ceil(len(rows)/64);warmup=max(1,int(total*.05));step=0;history=[];began=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();model.block.eval();order=rng.permutation(rows);losses=[];started=time.perf_counter()
        for first in range(0,len(order),64):
            group=order[first:first+64];optimizer.zero_grad(set_to_none=True)
            for at in range(0,len(group),16):
                selected=group[at:at+16];cls=vectors[torch.as_tensor(selected,device="cuda")]
                logits=model.classifier(model.dropout(model.pooler(cls[:,None,:])))
                if objective in ["forward_noise","residual_noise"]:
                    effective=logits if fixed_offset is None else logits+torch.from_numpy(fixed_offset[selected]).cuda()
                    loss=forward_noise_loss(torch,effective,torch.as_tensor(labels[selected],device="cuda"),eps)
                else:
                    target=torch.from_numpy(teacher_posterior(teacher[selected],labels[selected],eps)).cuda()
                    loss=-(target*torch.nn.functional.log_softmax(logits.float(),dim=-1)).sum(1).mean()
                if not torch.isfinite(loss):raise ValueError("冻结控制损失非有限。")
                (loss*len(selected)/len(group)).backward();losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            step+=1;factor=step/warmup if step<=warmup else max(0.,(total-step)/(total-warmup))
            optimizer.param_groups[0]["lr"]=rate*5*factor;optimizer.step()
        path=destination/f"epoch_{epoch}.pt";torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
        delta=max(float((v.detach().cpu()-initial[k]).abs().max()) for k,v in model.block.state_dict().items())
        if delta!=0:raise RuntimeError("固定编码器被修改。")
        history.append({"epoch":epoch,"seconds":time.perf_counter()-started,"training_loss":float(np.mean(losses))})
        print("固定编码器消融",epoch,history[-1],flush=True)
    model.eval();values=[]
    with torch.no_grad():
        for first in range(0,len(embedding),512):
            # 推理表示按被冻结的长度配置读取，分批搬运，避免另驻留一份256MB张量。
            inference_cls=torch.from_numpy(np.array(embedding[first:first+512])).cuda()
            value=model.classifier(model.dropout(model.pooler(inference_cls[:,None,:])))
            values.append(value.cpu().numpy())
    stem=f"neural_{stage}_classifier_only";np.savez_compressed(OUT/(stem+".npz"),id=controls["id"],logits=np.concatenate(values))
    store_cls(stem,np.asarray(embedding),path)
    write_json(destination/"training.json",{"stage":stage,"training_rows":len(rows),"epochs_completed":epochs,
        "planned_epochs":2,"learning_rate":rate,"classifier_lr_multiplier":5,"objective":objective,
        "effective_batch":64,"micro_batch":16,"encoder_block_updated":False,"encoder_parameter_max_change":0,
        "frozen_encoder_eval":True,"frozen_cls_source":str(source),"history":history,"seconds":time.perf_counter()-began,
        "frozen_training_cls_source":str(train_source),"training_max_length":128,"inference_max_length":max_length,
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)})
