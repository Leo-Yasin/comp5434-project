"""从已训练最后层快照保留更新后的CLS，供强正则融合对照使用。

只做前向计算，不重训编码层；先核对重算logits与原快照一致，再保存CLS。
数据放C盘，避免F盘不足；确认/最终和内部模型严格分开。
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
import gc
import json
import numpy as np
import torch
from .partial9 import load_top
from .partial9_protocol import OUT,FOLDER,run_name,tag,objective_suffix
from .partial9_cache import cached_batch
from .audit import sha256,write_json


def store_cls(stem,embedding,checkpoint,device="cuda"):
    """与同一次前向得到的logits绑定，后续可免去重复编码计算。"""
    root=FOLDER/"cls";root.mkdir(exist_ok=True);np.save(root/(stem+".npy"),embedding)
    write_json(root/(stem+".json"),{"checkpoint_sha256":sha256(checkpoint),
        "prediction_sha256":sha256(OUT/(stem+".npz")),"rows":len(embedding),"columns":768,"top_backend":device})


def valid_existing(stem,checkpoint):
    root=FOLDER/"cls";meta=root/(stem+".json")
    if not meta.exists() or not (root/(stem+".npy")).exists():return False
    record=json.loads(meta.read_text(encoding="utf-8"))
    if record["checkpoint_sha256"]!=sha256(checkpoint) or record["prediction_sha256"]!=sha256(OUT/(stem+".npz")):
        raise ValueError("CLS与快照/预测指纹不符。")
    value=np.load(root/(stem+".npy"),mmap_mode="r")
    if value.shape!=(record["rows"],768):raise ValueError("已有CLS形状不符。")
    return True


def inference_batch_size(folder,device):
    """只启用已经实测通过的CUDA前向批量；训练和CPU部署批量不变。"""
    audit=OUT/"gpu_prediction_batch_audit.json"
    if str(device).startswith("cuda") and audit.exists():
        proof=json.loads(audit.read_text(encoding="utf-8"))
        if proof["accepted"] and int(np.load(folder/"lengths.npy",mmap_mode="r").max())<=128:
            return int(proof["gpu_prediction_batch"])
    return 16


def extract(model,folder,rows=None,device="cuda",batch_size=None):
    """提取不带梯度的logits/CLS；较大CUDA批量必须先通过独立数值核验。"""
    lengths=np.load(folder/"lengths.npy",mmap_mode="r")
    batch_size=inference_batch_size(folder,device) if batch_size is None else batch_size
    rows=np.arange(len(lengths)) if rows is None else np.asarray(rows)
    destination=np.full(len(lengths),-1,np.int32);destination[rows]=np.arange(len(rows))
    order=rows[np.argsort(lengths[rows],kind="stable")]
    logits=np.empty((len(rows),5),np.float32);embedding=np.empty((len(rows),768),np.float32)
    model.eval()
    with torch.no_grad():
        for first in range(0,len(order),batch_size):
            selected=order[first:first+batch_size];hidden,mask=cached_batch(folder,selected,device)
            value,cls=model(hidden,mask);at=destination[selected]
            logits[at]=value.cpu().numpy();embedding[at]=cls.cpu().numpy()
    if not np.isfinite(logits).all() or not np.isfinite(embedding).all():raise ValueError("CLS提取数值非法。")
    if str(device).startswith("cuda"):torch.cuda.empty_cache()
    return logits,embedding


def one(stage,choice,classifier_only=False,device="cuda"):
    controls=np.load(OUT/"controls.npz");root=FOLDER/"cls";root.mkdir(exist_ok=True)
    path=FOLDER/run_name(stage,choice,classifier_only)/f"epoch_{choice['epochs']}.pt"
    positions=np.load(FOLDER/"long_positions.npy")
    if stage=="inner":
        stem=f"neural_inner_{tag(choice['learning_rate'])}_e{choice['epochs']}{objective_suffix(choice['objective'])}"
        if valid_existing(stem,path) and valid_existing(stem+"_long",path):
            print("复用同次前向已核验CLS",stem,flush=True);return
    else:
        stem=f"neural_{stage}"+("_classifier_only" if classifier_only else "")
        if valid_existing(stem,path):print("复用已核验CLS",stem,flush=True);return
    model=load_top(path).to(device).eval()
    if stage=="inner":
        rows=np.flatnonzero(controls["groups"]%100==0)
        stem=f"neural_inner_{tag(choice['learning_rate'])}_e{choice['epochs']}{objective_suffix(choice['objective'])}"
        resumed_short=valid_existing(stem,path)
        if resumed_short:
            # 短上下文完成、长上下文中断时保留已经核验的短表示。
            # 不重写短文件，避免已完成的内部融合缓存失去其输入SHA绑定。
            expected=np.load(OUT/(stem+".npz"));np.testing.assert_array_equal(expected["id"],controls["id"][rows])
            logits=expected["logits"].copy();cls=np.load(root/(stem+".npy")).copy()
        else:
            logits,cls=extract(model,FOLDER/"development",rows,device=device)
            expected=np.load(OUT/(stem+".npz"));np.testing.assert_array_equal(expected["id"],controls["id"][rows])
            np.testing.assert_allclose(logits,expected["logits"],atol=5e-5,rtol=0)
            store_cls(stem,cls,path,device)
        long_rows=np.flatnonzero(np.isin(positions,rows))
        long_logits,long_cls=extract(model,FOLDER/"long_development",long_rows,device=device)
        at=np.searchsorted(rows,positions[long_rows]);logits[at]=long_logits;cls[at]=long_cls
        expected=np.load(OUT/(stem+"_long.npz"));np.testing.assert_allclose(logits,expected["logits"],atol=5e-5,rtol=0)
        store_cls(stem+"_long",cls,path,"resumed_short_plus_"+device if resumed_short else device)
    else:
        stem=f"neural_{stage}"+("_classifier_only" if classifier_only else "")
        logits,cls=extract(model,FOLDER/"development",device=device)
        if choice["max_length"]==256:
            logits[positions],cls[positions]=extract(model,FOLDER/"long_development",device=device)
        expected=np.load(OUT/(stem+".npz"));np.testing.assert_array_equal(expected["id"],controls["id"])
        np.testing.assert_allclose(logits,expected["logits"],atol=5e-5,rtol=0)
        store_cls(stem,cls,path,device)
    print("更新CLS已提取并核对",stage,choice.get("objective"),choice["learning_rate"],choice["epochs"],flush=True)
    del model;gc.collect()
    if device=="cuda":torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["inner","confirm","final"],required=True)
    p.add_argument("--classifier-only",action="store_true")
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--objective",choices=["all","forward_noise","teacher_posterior","residual_noise"],default="all")
    args=p.parse_args();torch.set_num_threads(1)
    if args.device=="cuda":torch.cuda.set_per_process_memory_fraction(.8)
    torch.use_deterministic_algorithms(True)
    if args.stage=="inner":
        protocol=json.loads((OUT/"selection_protocol.json").read_text(encoding="utf-8"))
        for run in protocol["objective_runs"]:
            if args.objective!="all" and run["objective"]!=args.objective:continue
            for epoch in [1,2]:one("inner",{**run,"epochs":epoch,"max_length":256},device=args.device)
    else:
        choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
        if choice["route"]=="cls_head":one(args.stage,choice,args.classifier_only,args.device)


if __name__=="__main__":main()
