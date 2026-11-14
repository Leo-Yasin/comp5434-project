"""最后层拆分、缓存精度与本地GPU训练实测，全部样本只来自内部训练组。"""
import argparse
import copy
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from .audit import ROOT,write_json,sha256
from .partial9 import SOURCE,FOLDER,OUT,load_top,frozen_prefix
from .english_infer import make_backend
from .finetune_protocol import forward_noise_loss


def audit(cpu_only=False,max_length=128):
    torch.set_num_threads(1)
    frame=pd.read_csv(ROOT/"task2/outputs/round6_20261108/development_text.csv",keep_default_na=False)
    original=np.load(ROOT/"task2/outputs/round3_20261105/development.npz")
    index=np.load(ROOT/"task2/outputs/round6_20261108/selection.npz")["index"]
    groups=original["groups"][index];fit=(groups%100==0)&((groups//100)%5!=0)
    if max_length==256:fit &= np.load(FOLDER/"development/lengths.npy")==128
    n=128 if max_length==128 else 32
    candidates=np.flatnonzero(fit);rows=candidates[np.linspace(0,len(candidates)-1,n,dtype=int)]
    tokenizer=AutoTokenizer.from_pretrained(SOURCE/"tokenizer_spm",use_fast=False,local_files_only=True)
    encoded=tokenizer(frame.comment.iloc[rows].astype(str).tolist(),padding="max_length",truncation=True,max_length=max_length,return_tensors="np")
    suffix="" if max_length==128 else "_256"
    feed={k:encoded[k].astype(np.int64) for k in ["input_ids","attention_mask"]}
    if cpu_only:
        import openvino as ov
        full,_=make_backend(SOURCE/"model.onnx",4,"openvino")
        reference=[];refcls=[]
        audit_batch=16 if max_length==128 else 4
        for first in range(0,n,audit_batch):
            batch={k:v[first:first+audit_batch] for k,v in feed.items()}
            expected,cls=full(batch)
            reference.append(expected.copy());refcls.append(cls.copy())
            print("整网参考",max_length,first+len(expected),"/",n,flush=True)
        # 串行释放整网实现，再加载前缀图，避免同时占用两套模型/激活内存。
        del full;gc.collect()
        core=ov.Core();prefix=core.compile_model(str(FOLDER/"prefix.onnx"),"CPU",{
            "INFERENCE_PRECISION_HINT":ov.Type.f32,"INFERENCE_NUM_THREADS":4,"NUM_STREAMS":1})
        top=load_top().eval();exact=[];compressed=[];newcls=[];cached=[]
        for first in range(0,n,audit_batch):
            batch={k:v[first:first+audit_batch] for k,v in feed.items()}
            hidden=prefix(batch)[prefix.output(0)].copy()
            with torch.no_grad():
                logits,_=top(torch.from_numpy(hidden),torch.from_numpy(batch["attention_mask"]).bool())
                exact.append(logits.numpy())
                # 有效token压缩到FP16，补零padding；分类误差门槛在此实际检查。
                hidden=hidden.astype(np.float16).astype(np.float32)
                hidden*=batch["attention_mask"][:,:,None]
                logits,cls=top(torch.from_numpy(hidden),torch.from_numpy(batch["attention_mask"]).bool())
                compressed.append(logits.numpy());newcls.append(cls.numpy());cached.append(hidden)
            print("前缀数值核对",max_length,first+len(hidden),"/",n,flush=True)
        ref=np.concatenate(reference);fp=np.concatenate(exact);comp=np.concatenate(compressed)
        refcls=np.concatenate(refcls);newcls=np.concatenate(newcls)
        record={"rows":n,"max_length":max_length,"course_labels_used":False,"fp32_max_logit_error":float(np.abs(ref-fp).max()),
            "cache_fp16_max_logit_error":float(np.abs(ref-comp).max()),"cache_fp16_logit_mae":float(np.abs(ref-comp).mean()),
            "cache_fp16_argmax_agreement":float(np.mean(ref.argmax(1)==comp.argmax(1))),
            "cache_cls_relative_rmse":float(np.sqrt(np.mean((newcls-refcls)**2)/np.mean(refcls**2)))}
        if max_length==256:
            broad={k:v[:16] for k,v in feed.items()}
            with torch.no_grad():
                value=prefix(broad)[prefix.output(0)].copy()
                logits,_=top(torch.from_numpy(value),torch.from_numpy(broad["attention_mask"]).bool())
            record["batch16_max_logit_error"]=float(np.abs(logits.numpy()-ref[:16]).max())
        record["accepted"]=record["fp32_max_logit_error"]<.001 and record["cache_fp16_logit_mae"]<.01 and record["cache_fp16_argmax_agreement"]>=.99 and record["cache_cls_relative_rmse"]<.005
        if max_length==256:record["accepted"] &= record["batch16_max_logit_error"]<.001
        np.savez(FOLDER/f"audit_batch{suffix}.npz",hidden=np.concatenate(cached),mask=feed["attention_mask"],reference=ref)
        write_json(OUT/f"split_precision_audit{suffix}.json",record);print("拆分与缓存数值核验",record,flush=True)
        if not record["accepted"]:raise RuntimeError("拆分精度不通过，不准训练。")
        return
    if not torch.cuda.is_available():raise RuntimeError("CUDA不可用。")
    reference=np.load(FOLDER/f"audit_batch{suffix}.npz");record={"torch":torch.__version__,"device":torch.cuda.get_device_name(),
        "capability":list(torch.cuda.get_device_capability()),"course_labels_used":False,"training":[]}
    # 原前缀GPU推理和CPU图输出应一致；benchmark固定128token，不能用极短句夸大吞吐。
    prefix=frozen_prefix("cuda");started=time.perf_counter();parts=[]
    batch_size=8 if max_length==128 else 4
    for first in range(0,n,batch_size):
        with torch.no_grad():
            result=prefix(input_ids=torch.from_numpy(feed["input_ids"][first:first+batch_size]).cuda(),
                attention_mask=torch.from_numpy(feed["attention_mask"][first:first+batch_size]).cuda(),output_hidden_states=False).last_hidden_state
        parts.append(result.cpu().numpy())
    torch.cuda.synchronize();record.update(prefix_seconds=time.perf_counter()-started,prefix_rows=n,max_length=max_length,prefix_batch=batch_size)
    gpu_hidden=np.concatenate(parts);mask=reference["mask"][:,:,None]
    record["gpu_prefix_vs_fp16_cache_relative_rmse"]=float(np.sqrt(np.sum(((gpu_hidden-reference["hidden"])*mask)**2)/np.sum((gpu_hidden*mask)**2)))
    del prefix;gc.collect();torch.cuda.empty_cache()
    initial=load_top().eval();outputs=[]
    with torch.no_grad():
        for first in range(0,n,16):
            hidden=gpu_hidden[first:first+16].astype(np.float16).astype(np.float32)
            hidden*=mask[first:first+16].astype(np.float32)
            outputs.append(initial(torch.from_numpy(hidden),torch.from_numpy(reference["mask"][first:first+16]).bool())[0].numpy())
    outputs=np.concatenate(outputs)
    record["gpu_cached_complete_logit_mae"]=float(np.abs(outputs-reference["reference"]).mean())
    record["gpu_cached_complete_argmax_agreement"]=float(np.mean(outputs.argmax(1)==reference["reference"].argmax(1)))
    del initial
    # 256只用于推理上下文，正式训练始终128；无需以256训练显存决定其可用性。
    record["training_token_limit"]=128
    for size in ([8,16] if max_length==128 else []):
        model=load_top().cuda().train();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
        hidden=torch.from_numpy(reference["hidden"][:size]).cuda();attention=torch.from_numpy(reference["mask"][:size]).cuda().bool()
        original_weight=model.block.attention.self.query_proj.weight.detach().clone()
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        for _ in range(6):
            optimizer.zero_grad(set_to_none=True);logits,_=model(hidden,attention)
            loss=forward_noise_loss(torch,logits,torch.arange(size,device="cuda")%5,.25165131324376977)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        torch.cuda.synchronize()
        record["training"].append({"batch":size,"rows_per_second":size*6/(time.perf_counter()-started),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"last_layer_weight_updated":bool(not torch.equal(original_weight,model.block.attention.self.query_proj.weight))})
        del model,optimizer,hidden,attention;gc.collect();torch.cuda.empty_cache()
    record["accepted"]=record["gpu_prefix_vs_fp16_cache_relative_rmse"]<.002 and all(v["last_layer_weight_updated"] for v in record["training"]) and record["gpu_cached_complete_logit_mae"]<.01 and record["gpu_cached_complete_argmax_agreement"]>=.99
    write_json(OUT/f"local_gpu_audit{suffix}.json",record);print("本地GPU实测",record,flush=True)
    if not record["accepted"]:raise RuntimeError("本地GPU核验未通过。")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--cpu-only",action="store_true");p.add_argument("--max-length",type=int,choices=[128,256],default=128)
    args=p.parse_args();audit(args.cpu_only,args.max_length)
