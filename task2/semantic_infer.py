"""冻结的 BERT 五星分布推理，运行于单独的 Python 3.12 + ONNX 环境。

不需要训练标签，不调用外部收费推理服务。标题和正文构成一对输入，保留词序；
输出五类 logits，供自主噪声校正融合器使用。支持分块检查点和严格输入指纹校验。
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import onnxruntime as ort
from tokenizers import Tokenizer


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    began = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=args.max_length, strategy="longest_first")
    # 固定长度便于 DirectML 编译复用；CPU 也保持相同输入，避免实现差异。
    tokenizer.enable_padding(length=args.max_length, pad_id=0, pad_token="[PAD]")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False  # DirectML 官方要求关闭内存模式优化。
    options.log_severity_level = 3
    # 本机 MX450 的静态大图融合出现过驱动重置；保持动态形状，逐批裁掉无效 padding。
    if args.provider == "cpu" and not args.dynamic_length:
        options.add_free_dimension_override_by_name("batch_size", args.batch_size)
        options.add_free_dimension_override_by_name("sequence_length", args.max_length)
    if args.provider == "dml":
        model_file = args.model / "onnx/model_fp16.onnx"
        providers = [("DmlExecutionProvider", {"device_id": args.device}), "CPUExecutionProvider"]
    else:
        model_file = args.model / "onnx/model_quantized.onnx"
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_file), sess_options=options, providers=providers)
    print("语义推理后端", session.get_providers(), "输入", [(i.name, i.shape) for i in session.get_inputs()], flush=True)
    # 保留字面量 None；准备阶段已统一缺失值，这里只把空字符串保留为文本。
    frame = pd.read_csv(args.input, keep_default_na=False)
    if args.limit:
        frame = frame.iloc[:args.limit].copy()
    n = len(frame)
    fingerprint = {"input_sha256": file_hash(args.input), "model_sha256": file_hash(model_file),
                   "max_length": args.max_length, "provider": args.provider, "rows": n,
                   "batch_size": args.batch_size, "device": args.device,
                   "dynamic_length": args.dynamic_length}
    # 排序仅依赖输入长度，输出仍写回原行。缩短大多数短评论的无效计算。
    order = np.argsort((frame.title.str.len()+frame.comment.str.len()).to_numpy(), kind="stable") if args.dynamic_length else np.arange(n)
    state_path = args.output.with_suffix(".state.json")
    logits_path = args.output.with_suffix(".logits.npy")
    start = 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint:
            raise ValueError("断点参数或输入内容已改变，不能混用旧语义预测。")
        start = state["completed"]
        logits = np.lib.format.open_memmap(logits_path, mode="r+")
    else:
        logits = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(n, 5))
    input_names = {i.name for i in session.get_inputs()}
    inference_began = time.perf_counter()
    checkpoint = start
    for at in range(start, n, args.batch_size):
        indices = order[at:at+args.batch_size]
        batch = frame.iloc[indices]
        pairs = list(zip(batch.title.astype(str).tolist(), batch.comment.astype(str).tolist()))
        actual = len(pairs)
        # 最后不足一批时补输入，预测后裁回原长度，保持 GPU 形状固定。
        pairs += [pairs[-1]] * (args.batch_size-actual)
        encoded = tokenizer.encode_batch(pairs)
        feed = {"input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.asarray([e.type_ids for e in encoded], dtype=np.int64)}
        if args.dynamic_length:
            # 长度向上取到8的倍数，便于矩阵运算；attention_mask 保证 padding 不参与语义。
            length = min(args.max_length, max(8, int(np.ceil(feed["attention_mask"].sum(axis=1).max()/8)*8)))
            feed = {key: value[:, :length] for key, value in feed.items()}
        prediction = session.run(None, {k:v for k,v in feed.items() if k in input_names})[0]
        if prediction.shape != (args.batch_size, 5) or not np.isfinite(prediction).all():
            raise ValueError("外部模型输出维度或数值异常。")
        completed = at + actual
        logits[indices] = prediction[:actual]
        if completed-checkpoint >= 512 or completed == n:
            logits.flush()
            state_path.write_text(json.dumps({"fingerprint": fingerprint, "completed": completed,
                "last_run_inference_seconds": time.perf_counter()-inference_began}, indent=2), encoding="utf-8")
            rate = (completed-start)/max(time.perf_counter()-inference_began, 1e-9)
            print(f"语义推理 {completed:,}/{n:,}，{rate:.2f} 条/秒", flush=True)
            checkpoint = completed
    np.savez_compressed(args.output, id=frame.id.to_numpy(), logits=np.asarray(logits))
    timing = {**fingerprint, "complete_run_seconds": time.perf_counter()-began,
              "inference_seconds_this_run": time.perf_counter()-inference_began,
              "rows_resumed": start, "providers": session.get_providers()}
    args.output.with_suffix(".timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print("语义文件完成", str(args.output), timing, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=["cpu", "dml"], default="cpu")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dynamic-length", action="store_true")
    run(parser.parse_args())
