"""生成无标签测试标题的GPU冻结缓存，并记录明确的跨运行时精度/来源。

复用相同固定原英语权重，输出不含课程评分。CPU预测器只在输入、固定ONNX、
分词器、输出SHA以及已通过的CPU/GPU精度全部一致时才接受该缓存。
无缓存的新输入仍可完全使用CPU模型预测，不需要本文件或训练缓存。
"""
import json
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from task2.partial9 import source_model,SOURCE
from task2.partial9_protocol import FOLDER as CACHE
from task2.audit import ROOT,sha256,write_json
from pathlib import Path


def main():
    folder=Path.home()/".cache/comp5434-title10";out=ROOT/"task2/outputs/round10_20261114"
    if (folder/"test.npz").exists():raise FileExistsError("测试标题已有结果，不覆盖。")
    audit=json.loads((out/"gpu_fidelity.json").read_text(encoding="utf-8"))
    if not audit["accepted"]:raise ValueError("开发标题CPU/GPU实现核验未通过。")
    english=CACHE/"model/previous_model/english";input_path=folder/"test_title.csv"
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.9)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=source_model().cuda().eval();tokenizer=AutoTokenizer.from_pretrained(english/"tokenizer",use_fast=False,local_files_only=True,trust_remote_code=False)
    frame=pd.read_csv(input_path,keep_default_na=False)
    encoded=tokenizer(frame.comment.astype(str).tolist(),padding="max_length",truncation=True,max_length=128,return_tensors="np")
    length=encoded["attention_mask"].sum(1);order=np.argsort(length,kind="stable")
    logits=np.lib.format.open_memmap(folder/"test.logits.npy",mode="w+",dtype=np.float32,shape=(len(frame),5))
    cls=np.lib.format.open_memmap(folder/"test.embedding.npy",mode="w+",dtype=np.float32,shape=(len(frame),768))
    first=0;checkpoint=0;began=time.perf_counter()
    with torch.no_grad():
        while first<len(order):
            peek=order[first:first+32];maximum=int(length[peek].max());batch=32 if maximum<=32 else (16 if maximum<=64 else 8)
            rows=order[first:first+batch];n=max(8,int(np.ceil(length[rows].max()/8))*8)
            hidden=model.deberta(input_ids=torch.from_numpy(encoded["input_ids"][rows,:n]).cuda(),attention_mask=torch.from_numpy(encoded["attention_mask"][rows,:n]).cuda()).last_hidden_state
            value=model.classifier(model.dropout(model.pooler(hidden)))
            logits[rows]=value.cpu().numpy();cls[rows]=hidden[:,0].cpu().numpy();first+=len(rows)
            if first-checkpoint>=2048 or first==len(order):print("测试标题GPU编码",first,"/",len(frame),flush=True);checkpoint=first
    logits.flush();cls.flush()
    if not np.isfinite(logits).all() or not np.isfinite(cls).all():raise ValueError("测试标题结果非有限。")
    np.savez_compressed(folder/"test.npz",id=frame.id.to_numpy(),logits=np.asarray(logits))
    proof={"protocol":"verified_cuda_title_fp32_v1","completed":len(frame),"rows":len(frame),"labels_used":False,
        "source_revision":"06f020e0dbf909570eb886423bd3af256b855546","input_sha256":sha256(input_path),
        "reference_onnx_sha256":sha256(english/"model.onnx"),"tokenizer_sha256":{p.name:sha256(p) for p in (english/"tokenizer").iterdir() if p.is_file()},
        "max_length":128,"dtype":"float32","backend":"torch_cuda_fp32","torch":torch.__version__,"cpu_gpu_audit":audit,
        "npz_sha256":sha256(folder/"test.npz"),"logits_sha256":sha256(folder/"test.logits.npy"),"cls_sha256":sha256(folder/"test.embedding.npy")}
    write_json(folder/"test.state.json",proof)
    write_json(out/"test_gpu_timing.json",{"rows":len(frame),"seconds":time.perf_counter()-began,"rows_resumed":0,"backend":"torch_cuda_fp32","parameters_updated":False,"provenance":proof})
    print("测试标题GPU冻结缓存完成",len(frame),flush=True)


if __name__=="__main__":main()
