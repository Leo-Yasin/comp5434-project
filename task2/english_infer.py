"""冻结英语DeBERTa的本地CPU推理，保存五星logits和768维CLS。

输入仅正文，遵循作者教程的评论正文分类方式；标题信号仍由旧模型处理。
固定版本/批量/排序并校验断点，绝不读取标签。正式第八轮采用原FP32网络；
此前精度审计失败的INT8图不得替换正式网络。模型只做推理，不进行微调。
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer
from .audit import sha256,write_json


def make_backend(model,threads,backend):
    """OpenVINO明确请求FP32；不把自动低精度结果冒充原FP32网络。"""
    if backend=="openvino":
        import openvino as ov
        core=ov.Core();network=core.read_model(str(model))
        compiled=core.compile_model(network,"CPU",{"INFERENCE_PRECISION_HINT":ov.Type.f32,
            "INFERENCE_NUM_THREADS":threads,"NUM_STREAMS":1})
        names=[compiled.output("logits"),compiled.output("cls_embedding")]
        def infer(feed):
            result=compiled(feed)
            return [result[name] for name in names]
        return infer,ov.__version__
    options=ort.SessionOptions();options.intra_op_num_threads=threads
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.enable_mem_pattern=False
    session=ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
    required={"logits","cls_embedding"}
    if not required.issubset({v.name for v in session.get_outputs()}):raise ValueError("英语模型输出不完整。")
    return lambda feed:session.run(["logits","cls_embedding"],feed),ort.__version__


def run(args):
    started=time.perf_counter();frame=pd.read_csv(args.input,keep_default_na=False)
    if args.limit:frame=frame.iloc[:args.limit]
    n=len(frame)
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,use_fast=False,local_files_only=True,trust_remote_code=False)
    infer,version=make_backend(args.model,args.threads,args.backend)
    tokenizer_hashes={p.name:sha256(p) for p in args.tokenizer.iterdir() if p.is_file()}
    fingerprint={"input_sha256":sha256(args.input),"model_sha256":sha256(args.model),"tokenizer_sha256":tokenizer_hashes,
        "rows":n,"batch_size":args.batch_size,"max_length":128,"threads":args.threads,"input_field":"comment", "schema":1,
        "backend":args.backend,"runtime_version":version}
    state=args.output.with_suffix(".state.json");ep=args.output.with_suffix(".embedding.npy");lp=args.output.with_suffix(".logits.npy")
    args.output.parent.mkdir(parents=True,exist_ok=True);first=0
    if state.exists():
        previous=json.loads(state.read_text(encoding="utf-8"))
        if previous["fingerprint"]!=fingerprint:raise ValueError("英语推理配置变化，拒绝混用断点。")
        first=previous["completed"];embeddings=np.lib.format.open_memmap(ep,mode="r+");logits=np.lib.format.open_memmap(lp,mode="r+")
    else:
        if args.output.exists():raise FileExistsError(args.output)
        embeddings=np.lib.format.open_memmap(ep,mode="w+",dtype=np.float32,shape=(n,768))
        logits=np.lib.format.open_memmap(lp,mode="w+",dtype=np.float32,shape=(n,5))
    order=np.argsort(frame.comment.str.len().to_numpy(),kind="stable")
    inference=time.perf_counter();checkpoint=first
    for begin in range(first,n,args.batch_size):
        selected=order[begin:begin+args.batch_size];texts=frame.comment.iloc[selected].astype(str).tolist();actual=len(texts)
        texts += [texts[-1]]*(args.batch_size-actual)
        encoded=tokenizer(texts,padding="max_length",truncation=True,max_length=128,return_tensors="np")
        feed={k:np.asarray(encoded[k],dtype=np.int64) for k in ["input_ids","attention_mask"]}
        length=min(128,max(8,int(np.ceil(feed["attention_mask"].sum(axis=1).max()/8))*8))
        feed={k:v[:,:length] for k,v in feed.items()}
        predicted,hidden=infer(feed)
        if predicted.shape!=(args.batch_size,5) or hidden.shape!=(args.batch_size,768) or not np.isfinite(predicted).all() or not np.isfinite(hidden).all():
            raise ValueError("英语网络输出数值/形状非法。")
        logits[selected]=predicted[:actual];embeddings[selected]=hidden[:actual]
        done=begin+actual
        if done-checkpoint>=1024 or done==n:
            logits.flush();embeddings.flush()
            temporary=state.with_suffix(".json.tmp")
            write_json(temporary,{"fingerprint":fingerprint,"completed":done});temporary.replace(state)
            print(f"英语CPU推理 {done:,}/{n:,}，{(done-first)/max(time.perf_counter()-inference,1e-8):.2f}条/秒",flush=True)
            checkpoint=done
    np.savez_compressed(args.output,id=frame.id.to_numpy(),logits=np.asarray(logits))
    write_json(args.output.with_suffix(".timing.json"),{**fingerprint,"seconds":time.perf_counter()-started,"rows_resumed":first,
        "inference_seconds":time.perf_counter()-inference})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ["model","tokenizer","input","output"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--threads",type=int,default=2);p.add_argument("--limit",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--backend",choices=["ort","openvino"],default="ort")
    run(p.parse_args())
