"""冻结完整英语模型提取标题CLS/logits；本机GPU，完全不读取课程评分。

先等待第九轮释放GPU，再用内部拟合行的16个标题核对CPU正式实现。
未通过数值核验则不生成可用于选型的完成文件。标题采用原SPM、128上限，
仍保留字面量None；短标题按实际长度拼批，网络参数始终FP32和eval。
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from .partial9 import source_model,SOURCE
from .partial9_protocol import OUT as PARENT
from .audit import sha256,write_json

# 编码环境只需Torch/Transformers，不导入包含SciPy监督优化器的数值模块。
OUT=PARENT.with_name("round10_20261114")
FOLDER=Path.home()/".cache/comp5434-title10"


def wait_parent():
    """只等待本次局部流水线释放GPU，不创建系统定时任务。"""
    started=time.monotonic();last=-1
    while True:
        path=PARENT/"completion_pipeline.json"
        try:state=json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError,json.JSONDecodeError):state={}
        if state.get("status")=="failed":raise RuntimeError("父轮流水线失败，先修复父轮，不竞争GPU。")
        if any(v["name"]=="测试集全新CPU前缀与适配输出" for v in state.get("steps",[])):return
        elapsed=int(time.monotonic()-started)
        if elapsed>3600:raise TimeoutError("父轮GPU释放等待超过一小时，停止本分支。")
        if elapsed//60!=last:print("标题分支等待父轮GPU释放",elapsed,"秒",flush=True);last=elapsed//60
        time.sleep(2)


def main(wait):
    if wait:wait_parent()
    if (FOLDER/"development.npz").exists():raise FileExistsError("标题编码已存在，不覆盖完成结果。")
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.9)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=source_model().cuda().eval();tokenizer=AutoTokenizer.from_pretrained(SOURCE/"tokenizer_spm",use_fast=False,local_files_only=True,trust_remote_code=False)
    frame=pd.read_csv(FOLDER/"development_title.csv",keep_default_na=False)
    encoded=tokenizer(frame.comment.astype(str).tolist(),padding="max_length",truncation=True,max_length=128,return_tensors="np")
    tokens=np.asarray(encoded["input_ids"],dtype=np.int64);lengths=encoded["attention_mask"].sum(1)
    def infer(rows):
        length=max(8,int(np.ceil(lengths[rows].max()/8))*8)
        ids=torch.from_numpy(tokens[rows,:length]).cuda()
        mask=torch.from_numpy(np.asarray(encoded["attention_mask"][rows,:length],dtype=np.int64)).cuda()
        with torch.no_grad():
            hidden=model.deberta(input_ids=ids,attention_mask=mask).last_hidden_state
            logits=model.classifier(model.dropout(model.pooler(hidden)))
        return logits.cpu().numpy(),hidden[:,0].cpu().numpy()
    rows=np.load(FOLDER/"audit_rows.npy");a,b=infer(rows)
    ref=np.load(FOLDER/"audit.npz");np.testing.assert_array_equal(ref["id"],frame.id.to_numpy()[rows])
    original=np.load(FOLDER/"audit.embedding.npy")
    logit_mae=float(np.abs(a-ref["logits"]).mean());relative=float(np.linalg.norm(b-original)/np.linalg.norm(original))
    agreement=float(np.mean(a.argmax(1)==ref["logits"].argmax(1)))
    proof={"rows":len(rows),"course_labels_used":False,"logit_mae":logit_mae,"cls_relative_rmse":relative,
        "argmax_agreement":agreement,"accepted":logit_mae<.001 and relative<.001 and agreement==1.,
        "source_revision":"06f020e0dbf909570eb886423bd3af256b855546","cpu_reference":"original FP32 ONNX/OpenVINO"}
    write_json(OUT/"gpu_fidelity.json",proof);print("标题GPU/CPU数值核验",proof,flush=True)
    if not proof["accepted"]:raise RuntimeError("标题实现数值核验失败。")
    logits=np.lib.format.open_memmap(FOLDER/"development.logits.npy",mode="w+",dtype=np.float32,shape=(len(frame),5))
    cls=np.lib.format.open_memmap(FOLDER/"development.embedding.npy",mode="w+",dtype=np.float32,shape=(len(frame),768))
    order=np.argsort(lengths,kind="stable");first=0;checkpoint=0;began=time.perf_counter()
    while first<len(order):
        # 长输入缩小批量，使2GB显存始终有余量；这是无标签执行调度，非模型超参数。
        peek=order[first:first+32];maximum=int(lengths[peek].max());batch=32 if maximum<=32 else (16 if maximum<=64 else 8)
        selected=order[first:first+batch];value,embedding=infer(selected)
        if not np.isfinite(value).all() or not np.isfinite(embedding).all():raise ValueError("标题编码出现非有限数值。")
        logits[selected]=value;cls[selected]=embedding;first+=len(selected)
        if first-checkpoint>=2048 or first==len(order):
            logits.flush();cls.flush();write_json(FOLDER/"development.state.json",{"completed":first,"rows":len(frame),"input_sha256":sha256(FOLDER/"development_title.csv")})
            print("冻结标题编码",first,"/",len(frame),"新增条/秒",round(first/(time.perf_counter()-began),2),flush=True);checkpoint=first
    np.savez_compressed(FOLDER/"development.npz",id=frame.id.to_numpy(),logits=np.asarray(logits))
    write_json(OUT/"encoding_timing.json",{"rows":len(frame),"seconds":time.perf_counter()-began,"rows_resumed":0,
        "max_length":128,"torch":torch.__version__,"parameters_updated":False,"input_sha256":sha256(FOLDER/"development_title.csv"),
        "logits_sha256":sha256(FOLDER/"development.npz"),"cls_sha256":sha256(FOLDER/"development.embedding.npy")})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--wait-parent",action="store_true");args=p.parse_args();main(args.wait_parent)
