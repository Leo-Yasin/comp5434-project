"""仅用内部训练输入检查修复量化的精度，不用任何课程评分挑量化方案。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer
from .audit import ROOT,write_json
from .english_prepare import FOLDER


def main():
    out=ROOT/"task2/outputs/round8_20261114"
    original=np.load(ROOT/"task2/outputs/round3_20261105/development.npz")
    index=np.load(ROOT/"task2/outputs/round6_20261108/selection.npz")["index"]
    groups=original["groups"][index]
    fit=(groups%np.uint64(100)==0)&((groups//np.uint64(100))%np.uint64(5)!=0)
    frame=pd.read_csv(ROOT/"task2/outputs/round6_20261108/development_text.csv",keep_default_na=False)
    # 与SmoothQuant激活校准的前128条分离，只验证实现近似，不用评分选模型。
    text=sorted(frame.loc[fit,"comment"].iloc[128:256].astype(str).tolist(),key=len)
    tokenizer=AutoTokenizer.from_pretrained(FOLDER/"tokenizer_spm",local_files_only=True,use_fast=False,trust_remote_code=False)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    floating=ort.InferenceSession(str(FOLDER/"model.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
    selected=json.loads((FOLDER/"quantization_choice.json").read_text(encoding="utf-8"))["selected"]["file"]
    quantized=ort.InferenceSession(str(FOLDER/selected),sess_options=options,providers=["CPUExecutionProvider"])
    fp=[];qp=[];fe=[];qe=[];started=time.perf_counter()
    for first in range(0,len(text),16):
        batch=text[first:first+16]
        encoded=tokenizer(batch,padding="max_length",truncation=True,max_length=128,return_tensors="np")
        feed={k:np.asarray(encoded[k],dtype=np.int64) for k in ["input_ids","attention_mask"]}
        length=min(128,max(8,int(np.ceil(feed["attention_mask"].sum(axis=1).max()/8))*8))
        feed={k:v[:,:length] for k,v in feed.items()}
        a,b=floating.run(None,feed);c,d=quantized.run(None,feed)
        fp.append(a);fe.append(b);qp.append(c);qe.append(d)
    fp=np.concatenate(fp);qp=np.concatenate(qp);fe=np.concatenate(fe);qe=np.concatenate(qe)
    agreement=float(np.mean(fp.argmax(axis=1)==qp.argmax(axis=1)))
    mae=float(np.mean(np.abs(fp-qp)));relative=float(np.sqrt(np.mean((fe-qe)**2)/np.mean(fe**2)))
    result={"rows":len(text),"only_internal_training_inputs":True,"labels_used":False,"calibration_disjoint":True,
        "model":selected,"batch_size":16,"sorted_dynamic_length":True,"argmax_agreement":agreement,
        "logit_mae":mae,"cls_relative_rmse":relative,"seconds":time.perf_counter()-started,
        "accepted":agreement>=.95 and mae<.2 and relative<.15}
    write_json(out/("english_quantization_audit_"+Path(selected).stem+".json"),result)
    write_json(out/"english_quantization_audit.json",result);print("真实输入量化审计",result,flush=True)
    if not result["accepted"]:raise RuntimeError("真实输入精度门槛未通过，不能继续使用此量化。")


if __name__=="__main__":main()
