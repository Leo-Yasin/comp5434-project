"""第六轮冻结网络推理：同时返回五星logits和768维CLS，带严格断点校验。

保持和第五轮一致的WordPiece成对输入、长度排序、CPU量化、批量16及128上限。
只增加输出，不执行梯度或修改网络权重。CLS存为float32，避免额外半精度压缩误差。
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import numpy as np
import pandas as pd
import onnxruntime as ort
from tokenizers import Tokenizer

from .semantic_infer import file_hash


def run(args):
    began=time.perf_counter()
    frame=pd.read_csv(args.input,keep_default_na=False)
    if args.limit:frame=frame.iloc[:args.limit].copy()
    n=len(frame)
    tokenizer=Tokenizer.from_file(str(args.tokenizer))
    tokenizer.enable_truncation(max_length=128,strategy="longest_first")
    tokenizer.enable_padding(length=128,pad_id=0,pad_token="[PAD]")
    options=ort.SessionOptions();options.intra_op_num_threads=4
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.enable_mem_pattern=False
    session=ort.InferenceSession(str(args.encoder),sess_options=options,providers=["CPUExecutionProvider"])
    required={"logits","cls_embedding"}
    if not required.issubset({v.name for v in session.get_outputs()}):raise ValueError("中间表示输出缺失。")
    fingerprint={"input_sha256":file_hash(args.input),"encoder_sha256":file_hash(args.encoder),
        "tokenizer_sha256":file_hash(args.tokenizer),"rows":n,"batch_size":16,"max_length":128,
        "sorted_dynamic_length":True,"embedding_dtype":"float32","schema":1}
    state_path=args.output.with_suffix(".state.json")
    logits_path=args.output.with_suffix(".logits.npy")
    embedding_path=args.output.with_suffix(".embedding.npy")
    start=0
    if state_path.exists():
        saved=json.loads(state_path.read_text(encoding="utf-8"))
        if saved["fingerprint"]!=fingerprint:raise ValueError("输入/模型/配置变化，不能混用断点。")
        start=saved["completed"]
        logits=np.lib.format.open_memmap(logits_path,mode="r+")
        embedding=np.lib.format.open_memmap(embedding_path,mode="r+")
    else:
        if args.output.exists():raise FileExistsError(args.output)
        logits=np.lib.format.open_memmap(logits_path,mode="w+",dtype=np.float32,shape=(n,5))
        embedding=np.lib.format.open_memmap(embedding_path,mode="w+",dtype=np.float32,shape=(n,768))
    order=np.argsort((frame.title.str.len()+frame.comment.str.len()).to_numpy(),kind="stable")
    input_names={v.name for v in session.get_inputs()}
    inference=time.perf_counter();checkpoint=start
    for at in range(start,n,16):
        ids=order[at:at+16];batch=frame.iloc[ids]
        pairs=list(zip(batch.title.astype(str).tolist(),batch.comment.astype(str).tolist()))
        actual=len(pairs);pairs += [pairs[-1]]*(16-actual)
        encoded=tokenizer.encode_batch(pairs)
        feed={"input_ids":np.asarray([v.ids for v in encoded],np.int64),
              "attention_mask":np.asarray([v.attention_mask for v in encoded],np.int64),
              "token_type_ids":np.asarray([v.type_ids for v in encoded],np.int64)}
        length=min(128,max(8,int(np.ceil(feed["attention_mask"].sum(axis=1).max()/8)*8)))
        feed={k:v[:,:length] for k,v in feed.items() if k in input_names}
        lp,ep=session.run(["logits","cls_embedding"],feed)
        if lp.shape!=(16,5) or ep.shape!=(16,768) or not np.isfinite(ep).all() or not np.isfinite(lp).all():
            raise ValueError("CLS/logits形状或数值异常。")
        logits[ids]=lp[:actual];embedding[ids]=ep[:actual]
        completed=at+actual
        if completed-checkpoint>=1024 or completed==n:
            logits.flush();embedding.flush()
            state_path.write_text(json.dumps({"fingerprint":fingerprint,"completed":completed}),encoding="utf-8")
            print(f"CLS推理 {completed:,}/{n:,}，{(completed-start)/max(time.perf_counter()-inference,1e-8):.2f}条/秒",flush=True)
            checkpoint=completed
    # 大型CLS数组保留为可mmap的npy；索引小文件只存ID和logits，避免再复制一份全量CLS。
    np.savez_compressed(args.output,id=frame.id.to_numpy(),logits=np.asarray(logits))
    args.output.with_suffix(".timing.json").write_text(json.dumps({**fingerprint,
        "complete_seconds":time.perf_counter()-began,"inference_seconds":time.perf_counter()-inference,
        "rows_resumed":start,"embedding_file":embedding_path.name},indent=2),encoding="utf-8")
    print("CLS特征完成",str(args.output),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--encoder",type=Path,required=True)
    parser.add_argument("--tokenizer",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--limit",type=int,default=0)
    run(parser.parse_args())
