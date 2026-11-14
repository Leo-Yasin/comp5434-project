"""无标签前缀缓存：先分词，按真实token长度紧凑保存第11层输出。

只保存有效token，padding不成为key；最后一层没有卷积，补零padding不会
改变有效token的注意力。float16只压缩缓存，源/训练权重仍FP32，必须先核验误差。
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from .audit import sha256,write_json
from .partial9 import SOURCE,FOLDER,OUT,frozen_prefix


def tokenize(source,folder,tokenizer_folder=None,max_length=128):
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/"tokens.json"
    tokenizer_folder=tokenizer_folder or SOURCE/"tokenizer_spm"
    hashes={p.name:sha256(p) for p in tokenizer_folder.iterdir() if p.is_file()}
    fingerprint={"input_sha256":sha256(source),"tokenizer":hashes,"max_length":max_length,"input":"comment","schema":1}
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8"))["fingerprint"]!=fingerprint:raise ValueError("分词输入变化。")
        return
    frame=pd.read_csv(source,keep_default_na=False)
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_folder,use_fast=False,local_files_only=True,trust_remote_code=False)
    ids=np.lib.format.open_memmap(folder/"tokens.npy",mode="w+",dtype=np.int32,shape=(len(frame),max_length))
    lengths=np.zeros(len(frame),np.uint16);began=time.perf_counter()
    for first in range(0,len(frame),256):
        texts=frame.comment.iloc[first:first+256].astype(str).tolist()
        encoded=tokenizer(texts,padding="max_length",truncation=True,max_length=max_length,return_tensors="np")
        length=encoded["attention_mask"].sum(1)
        np.testing.assert_array_equal(encoded["attention_mask"],np.arange(max_length)[None,:]<length[:,None])
        ids[first:first+len(texts)]=encoded["input_ids"];lengths[first:first+len(texts)]=length
        if (first//256+1)%32==0:print("前缀分词",first+len(texts),"秒",time.perf_counter()-began,flush=True)
    ids.flush();np.save(folder/"lengths.npy",lengths);np.save(folder/"id.npy",frame.id.to_numpy())
    offsets=np.concatenate([[0],np.cumsum(lengths,dtype=np.int64)]);np.save(folder/"offsets.npy",offsets)
    write_json(target,{"fingerprint":fingerprint,"rows":len(frame),"valid_tokens":int(offsets[-1]),"seconds":time.perf_counter()-began})


@lru_cache(maxsize=4)
def read_cache(folder):
    """只读mmap复用句柄，避免每个微批反复打开大文件并解析npy头。"""
    return (np.load(folder/"lengths.npy",mmap_mode="r"),np.load(folder/"offsets.npy",mmap_mode="r"),
        np.load(folder/"hidden.npy",mmap_mode="r"))


def cached_batch(folder,rows,device="cpu"):
    """同一拼批函数用于训练和推理；只从紧凑数组恢复该批有效token。"""
    lengths,offset,values=read_cache(folder)
    length=int(np.ceil(lengths[rows].max()/8))*8
    hidden=np.zeros((len(rows),length,768),np.float32)
    for i,row in enumerate(rows):hidden[i,:lengths[row]]=values[offset[row]:offset[row+1]]
    mask=np.arange(length)[None,:]<lengths[rows,None]
    return torch.from_numpy(hidden).to(device),torch.from_numpy(mask).to(device)


def run(source,folder,backend,batch_size,prefix_path=None,tokenizer_folder=None,max_length=128):
    prefix_path=prefix_path or FOLDER/"prefix.onnx"
    tokenize(source,folder,tokenizer_folder,max_length);tokens=np.load(folder/"tokens.npy",mmap_mode="r")
    lengths=np.load(folder/"lengths.npy",mmap_mode="r");offset=np.load(folder/"offsets.npy",mmap_mode="r")
    meta=json.loads((folder/"tokens.json").read_text(encoding="utf-8"));n=len(tokens)
    if backend=="cuda":
        model=frozen_prefix("cuda");version=torch.__version__
        def infer(feed):
            with torch.no_grad():
                result=model(input_ids=torch.from_numpy(feed["input_ids"]).cuda(),
                    attention_mask=torch.from_numpy(feed["attention_mask"]).cuda(),output_hidden_states=False).last_hidden_state
                return result.cpu().numpy()
    else:
        import openvino as ov
        core=ov.Core();model=core.compile_model(str(prefix_path),"CPU",{
            "INFERENCE_PRECISION_HINT":ov.Type.f32,"INFERENCE_NUM_THREADS":4,"NUM_STREAMS":1})
        version=ov.__version__
        def infer(feed):return model(feed)[model.output(0)]
    fingerprint={"tokens":meta["fingerprint"],"prefix_sha256":sha256(prefix_path),
        "backend":backend,"version":version,"batch_size":batch_size,"dtype":"float16","schema":1}
    state=folder/"state.json";first=0;started=time.perf_counter()
    if state.exists():
        record=json.loads(state.read_text(encoding="utf-8"))
        if record["fingerprint"]!=fingerprint:raise ValueError("前缀推理配置变化，不能混合断点。")
        first=record["completed"];hidden=np.lib.format.open_memmap(folder/"hidden.npy",mode="r+")
    else:
        hidden=np.lib.format.open_memmap(folder/"hidden.npy",mode="w+",dtype=np.float16,shape=(int(offset[-1]),768))
    order=np.argsort(lengths,kind="stable");checkpoint=first
    for begin in range(first,n,batch_size):
        selected=order[begin:begin+batch_size];actual=len(selected)
        padded=np.pad(selected,(0,batch_size-actual),mode="edge")
        length=int(np.ceil(lengths[padded].max()/8))*8
        feed={"input_ids":np.asarray(tokens[padded,:length],dtype=np.int64),
            "attention_mask":np.asarray(np.arange(length)[None,:]<lengths[padded,None],dtype=np.int64)}
        result=infer(feed)
        if result.shape!=(batch_size,length,768) or not np.isfinite(result).all():raise ValueError("前缀数值/形状不合法。")
        for i,row in enumerate(selected):hidden[offset[row]:offset[row+1]]=result[i,:lengths[row]]
        completed=begin+actual
        if completed-checkpoint>=1024 or completed==n:
            hidden.flush();temp=state.with_suffix(".tmp")
            write_json(temp,{"fingerprint":fingerprint,"completed":completed});temp.replace(state)
            print(f"前11层缓存 {completed}/{n}，新增{(completed-first)/max(time.perf_counter()-started,1e-8):.2f}条/秒",flush=True)
            checkpoint=completed
    write_json(folder/"timing.json",{"rows":n,"rows_resumed":first,"seconds":time.perf_counter()-started,
        "backend":backend,"cached_tokens":int(offset[-1]),"bytes":int(hidden.nbytes),"fingerprint":fingerprint})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--folder",type=Path,required=True)
    p.add_argument("--backend",choices=["cuda","openvino"],required=True);p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--max-length",type=int,choices=[128,256],default=128)
    args=p.parse_args();torch.set_num_threads(1);run(args.input,args.folder,args.backend,args.batch_size,max_length=args.max_length)
