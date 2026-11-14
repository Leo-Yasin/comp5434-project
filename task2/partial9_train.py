"""本地最后层适配：FP32、固定噪声似然、只训练被允许的标签行。

前11层的有效token缓存完全无标签；这里更新最后一个Transformer block、
原pooler和五星classifier。GPU只放这部分，避免2GB显存容纳整网优化器。
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
import json
import math
import time
import numpy as np
import torch
from .partial9 import FOLDER,OUT,load_top
from .partial9_cache import cached_batch
from .finetune_protocol import forward_noise_loss
from .audit import write_json
from .partial9_protocol import tag,fit_mask,teacher_posterior,objective_suffix,run_name


def predict(model,folder,batch_size=None,device="cuda",rows=None):
    """按token长度稳定排序，统一拼批；不读取任何评分标签。"""
    from .partial9_cls import inference_batch_size
    batch_size=inference_batch_size(folder,device) if batch_size is None else batch_size
    lengths=np.load(folder/"lengths.npy",mmap_mode="r")
    requested=np.arange(len(lengths)) if rows is None else np.asarray(rows,dtype=np.int64)
    order=requested[np.argsort(lengths[requested],kind="stable")]
    result=np.empty((len(lengths),5),np.float32);model.eval()
    with torch.no_grad():
        for first in range(0,len(order),batch_size):
            rows=order[first:first+batch_size];hidden,mask=cached_batch(folder,rows,device)
            logits,_=model(hidden,mask);value=logits.cpu().numpy()
            if not np.isfinite(value).all():raise ValueError("适配网络预测非有限。")
            result[rows]=value
    output=result[requested]
    if str(device).startswith("cuda"):torch.cuda.empty_cache()
    return output


def train(stage,rate,epochs=2,freeze_block=False,eval_max_length=128,objective="forward_noise"):
    torch.set_num_threads(1);torch.manual_seed(5434);torch.cuda.manual_seed_all(5434)
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    controls=np.load(OUT/"controls.npz");folder=FOLDER/"development"
    np.testing.assert_array_equal(np.load(folder/"id.npy"),controls["id"])
    if json.loads((folder/"state.json").read_text(encoding="utf-8"))["completed"]!=len(controls["id"]):
        raise RuntimeError("前缀尚未完成，禁止使用未填满的缓存。")
    contexts=json.loads((OUT/"selection_protocol.json").read_text(encoding="utf-8"))["inference_max_lengths"]
    if (stage=="inner" and 256 in contexts) or eval_max_length==256:
        positions=np.load(FOLDER/"long_positions.npy");long_folder=FOLDER/"long_development"
        np.testing.assert_array_equal(np.load(long_folder/"id.npy"),controls["id"][positions])
        if json.loads((long_folder/"state.json").read_text(encoding="utf-8"))["completed"]!=len(positions):
            raise RuntimeError("长输入前缀尚未完成。")
    rows=np.flatnonzero(fit_mask(controls["groups"],stage));labels=controls["y"].astype(np.int64)-1
    epsilon=float(controls["epsilon"]);model=load_top().cuda()
    initial_block={k:v.detach().cpu().clone() for k,v in model.block.state_dict().items()}
    if freeze_block:model.block.requires_grad_(False)
    suffix="_classifier_only" if freeze_block else ""
    destination=FOLDER/run_name(stage,{"learning_rate":rate,"objective":objective},freeze_block);destination.mkdir(exist_ok=False)
    teacher=None
    if objective=="teacher_posterior":
        teacher=np.zeros_like(controls[stage+"_offset"],dtype=np.float64)
        raw=controls[stage+"_offset"][rows];value=np.exp(raw-raw.max(1,keepdims=True));value/=value.sum(1,keepdims=True)
        teacher[rows]=value
    fixed_offset=None
    if objective=="residual_noise":
        from .audit import ROOT
        original=np.load(ROOT/"task2/outputs/round8_20261114/english_development.npz")
        np.testing.assert_array_equal(original["id"],controls["id"])
        fixed_offset=(controls[stage+"_offset"]-original["logits"]).astype(np.float32)
    parameter_groups=[{"params":list(model.block.parameters()),"lr":rate},
        {"params":list(model.pooler.parameters())+list(model.classifier.parameters()),"lr":rate*5}]
    optimizer=torch.optim.AdamW(parameter_groups,weight_decay=.01,foreach=False)
    # 内部统一计划两轮，选中第一轮时，确认/最终仍使用两轮调度的前半段。
    # 这样快照1的学习率轨迹不会因早停后重训而悄悄改变。
    total_steps=2*math.ceil(len(rows)/64);warmup=max(1,int(total_steps*.05))
    rng=np.random.default_rng(5434);step=0;history=[];started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();order=rng.permutation(rows);losses=[];epoch_start=time.perf_counter()
        for first in range(0,len(order),64):
            group=order[first:first+64];optimizer.zero_grad(set_to_none=True)
            for at in range(0,len(group),16):
                selected=group[at:at+16];hidden,mask=cached_batch(folder,selected,"cuda")
                logits,_=model(hidden,mask)
                if objective in ["forward_noise","residual_noise"]:
                    effective=logits if fixed_offset is None else logits+torch.from_numpy(fixed_offset[selected]).cuda()
                    loss=forward_noise_loss(torch,effective,torch.as_tensor(labels[selected],device="cuda"),epsilon)
                else:
                    # 固定教师来自相应阶段第八轮；只构造当前拟合行的标签后验，确认标签不进入计算。
                    target=teacher_posterior(teacher[selected],labels[selected],epsilon)
                    loss=-(torch.from_numpy(target).cuda()*torch.nn.functional.log_softmax(logits.float(),dim=-1)).sum(1).mean()
                if not torch.isfinite(loss):raise ValueError("训练损失非有限。")
                (loss*len(selected)/len(group)).backward();losses.append(float(loss.detach()))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError("训练梯度非有限。")
            step+=1
            factor=step/warmup if step<=warmup else max(0.,(total_steps-step)/(total_steps-warmup))
            for group_,base in zip(optimizer.param_groups,[rate,rate*5]):group_["lr"]=base*factor
            optimizer.step()
            if step%50==0:print("最后层训练",stage,tag(rate),objective,"epoch",epoch,"rows",min(first+64,len(rows)),"loss",float(np.mean(losses[-40:])),flush=True)
        path=destination/f"epoch_{epoch}.pt"
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
        block_delta=max(float((v.detach().cpu()-initial_block[k]).abs().max()) for k,v in model.block.state_dict().items())
        if freeze_block and block_delta!=0:raise RuntimeError("冻结消融中的编码层权重发生改变。")
        if not freeze_block and block_delta==0:raise RuntimeError("编码层没有实际更新。")
        record={"epoch":epoch,"seconds":time.perf_counter()-epoch_start,"training_loss":float(np.mean(losses)),"steps":step}
        history.append(record);write_json(destination/"training.json",{"stage":stage,"training_rows":len(rows),
            "learning_rate":rate,"classifier_lr_multiplier":5,"planned_epochs":2,"epochs_completed":epoch,
            "effective_batch":64,"micro_batch":16,"epsilon":epsilon,"seed":5434,"history":history,
            "objective":objective,"teacher_training_predictions_cross_fitted":False if teacher is not None else None,
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "encoder_block_updated":not freeze_block,"encoder_parameter_max_change":block_delta,
            "seconds":time.perf_counter()-started})
        print("最后层快照",stage,tag(rate),record,flush=True)
        if stage=="inner":
            # 内部快照只输出41705条内部训练/检查行，确认行在本阶段不做适配评分。
            active=np.flatnonzero(controls["groups"]%100==0)
            from .partial9_cls import extract,store_cls
            values,embedding=extract(model,folder,rows=active)
            stem=f"neural_{stage}_{tag(rate)}_e{epoch}{objective_suffix(objective)}"
            np.savez_compressed(OUT/(stem+".npz"),id=controls["id"][active],logits=values)
            store_cls(stem,embedding,path)
            if 256 in contexts:
                positions=np.load(FOLDER/"long_positions.npy")
                long_active=np.flatnonzero(np.isin(positions,active))
                at=np.searchsorted(active,positions[long_active])
                values[at],embedding[at]=extract(model,FOLDER/"long_development",rows=long_active)
                np.savez_compressed(OUT/(stem+"_long.npz"),id=controls["id"][active],logits=values)
                store_cls(stem+"_long",embedding,path)
    if stage!="inner":
        selected=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
        if selected["route"]=="cls_head":
            from .partial9_cls import extract,store_cls
            values,embedding=extract(model,folder)
            if eval_max_length==256:
                values[np.load(FOLDER/"long_positions.npy")],embedding[np.load(FOLDER/"long_positions.npy")]=extract(model,FOLDER/"long_development")
            stem=f"neural_{stage}{suffix}";np.savez_compressed(OUT/(stem+".npz"),id=controls["id"],logits=values)
            store_cls(stem,embedding,path)
        else:
            values=predict(model,folder)
            if eval_max_length==256:values[np.load(FOLDER/"long_positions.npy")]=predict(model,FOLDER/"long_development")
            np.savez_compressed(OUT/f"neural_{stage}{suffix}.npz",id=controls["id"],logits=values)


def unadapted_long():
    """固定未微调的256token对照，用于区分上下文变化和权重适配的作用。"""
    from .audit import ROOT
    controls=np.load(OUT/"controls.npz");positions=np.load(FOLDER/"long_positions.npy")
    saved=np.load(ROOT/"task2/outputs/round8_20261114/english_development.npz")
    np.testing.assert_array_equal(saved["id"],controls["id"])
    from .partial9_cls import extract
    values=saved["logits"].copy();model=load_top().cuda().eval()
    values[positions],long_cls=extract(model,FOLDER/"long_development")
    embedding=np.load(ROOT/"task2/outputs/round8_20261114/english_development.embedding.npy",mmap_mode="r").copy()
    embedding[positions]=long_cls;(FOLDER/"cls").mkdir(exist_ok=True)
    np.save(FOLDER/"cls/neural_unadapted_long.npy",embedding)
    np.savez_compressed(OUT/"neural_unadapted_long.npz",id=controls["id"],logits=values)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["inner","confirm","final","unadapted"],required=True)
    p.add_argument("--learning-rate",type=float);p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--freeze-block",action="store_true",help="仅用于固定确认消融，不参与选型。")
    p.add_argument("--eval-max-length",type=int,choices=[128,256],default=128)
    p.add_argument("--objective",choices=["forward_noise","teacher_posterior","residual_noise"],default="forward_noise")
    args=p.parse_args()
    if args.stage=="unadapted":torch.set_num_threads(1);unadapted_long()
    else:
        if args.learning_rate not in [1e-5,3e-5] or args.epochs not in [1,2]:raise ValueError("训练参数不在预登记范围。")
        if args.objective=="teacher_posterior" and args.learning_rate!=1e-5:raise ValueError("该后验学习率未登记。")
        if args.objective=="residual_noise" and args.learning_rate!=3e-5:raise ValueError("该残差学习率未登记。")
        if args.freeze_block:
            from .partial9_head_control import train_control
            train_control(args.stage,args.learning_rate,args.epochs,args.eval_max_length,args.objective)
        else:train(args.stage,args.learning_rate,args.epochs,False,args.eval_max_length,args.objective)
