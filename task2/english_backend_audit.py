"""比较OpenVINO FP32和已核验ORT FP32，仅用输入和原网络输出，不使用课程评分。"""
import time
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from .audit import ROOT,write_json
from .english_prepare import FOLDER
from .english_infer import make_backend


def main():
    data=np.load(ROOT/"task2/outputs/round3_20261105/development.npz")
    index=np.load(ROOT/"task2/outputs/round6_20261108/selection.npz")["index"]
    groups=data["groups"][index]
    fit=(groups%np.uint64(100)==0)&((groups//np.uint64(100))%np.uint64(5)!=0)
    frame=pd.read_csv(ROOT/"task2/outputs/round6_20261108/development_text.csv",keep_default_na=False)
    text=sorted(frame.loc[fit,"comment"].iloc[128:256].astype(str).tolist(),key=len)
    tokenizer=AutoTokenizer.from_pretrained(FOLDER/"tokenizer_spm",local_files_only=True,use_fast=False,trust_remote_code=False)
    original,_=make_backend(FOLDER/"model.onnx",2,"ort")
    alternative,version=make_backend(FOLDER/"model.onnx",4,"openvino")
    a=[];b=[];ae=[];be=[];started=time.perf_counter()
    for first in range(0,len(text),16):
        encoded=tokenizer(text[first:first+16],padding="max_length",truncation=True,max_length=128,return_tensors="np")
        feed={k:np.asarray(encoded[k],dtype=np.int64) for k in ["input_ids","attention_mask"]}
        length=min(128,max(8,int(np.ceil(feed["attention_mask"].sum(axis=1).max()/8))*8));feed={k:v[:,:length] for k,v in feed.items()}
        x,y=original(feed);u,v=alternative(feed);a.append(x.copy());ae.append(y.copy());b.append(u.copy());be.append(v.copy())
    a=np.concatenate(a);b=np.concatenate(b);ae=np.concatenate(ae);be=np.concatenate(be)
    result={"rows":128,"labels_used":False,"backend":"openvino","version":version,"precision":"f32",
        "argmax_agreement":float(np.mean(a.argmax(1)==b.argmax(1))),"max_logit_error":float(np.abs(a-b).max()),
        "logit_mae":float(np.abs(a-b).mean()),"cls_relative_rmse":float(np.sqrt(np.mean((ae-be)**2)/np.mean(ae**2))),
        "seconds":time.perf_counter()-started}
    result["accepted"]=result["argmax_agreement"]>=.99 and result["logit_mae"]<.001 and result["cls_relative_rmse"]<.002
    write_json(ROOT/"task2/outputs/round8_20261114/english_fp32_backend_audit.json",result)
    print("英语FP32后端核对",result,flush=True)
    if not result["accepted"]:raise RuntimeError("FP32后端不一致，拒绝使用。")


if __name__=="__main__":main()
