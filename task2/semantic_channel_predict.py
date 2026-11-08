"""扩展通道候选预测：复用同一个语义模型包，明确使用自主噪声通道分类器。

只有额外冻结审计达到预登记条件才允许本轮交付脚本生成此候选。
语义NPZ可以来自独立原始CSV推理，但必须严格校验id和本轮固定推理配置。
"""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from .audit import ROOT, CSV_OPTIONS, write_json, sha256
from .semantic_predict import inherited_from_frame
from .semantic_fusion import semantic_features
from .noise_softmax import predict_channel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--semantic-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    began = time.perf_counter()
    # 审计许可随模型包交付，预测不依赖实验目录或原训练缓存。
    audit = json.loads((args.model/"channel_candidate.json").read_text(encoding="utf-8"))
    if not audit["release_channel_as_additional_candidate"]:
        raise RuntimeError("扩展审计未达到登记条件，不发布事后挑选的通道候选。")
    timing = json.loads(args.semantic_predictions.with_suffix(".timing.json").read_text(encoding="utf-8"))
    if timing["provider"] != "cpu" or timing["batch_size"] != 16 or timing["max_length"] != 128 or not timing["dynamic_length"]:
        raise ValueError("通道候选的语义配置不符。")
    frame = pd.read_csv(args.test, **CSV_OPTIONS)
    buffer = io.StringIO()
    frame[["id", "title", "comment"]].fillna("").to_csv(buffer, index=False)
    if hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest() != timing["input_sha256"]:
        raise ValueError("语义文件的文本内容与本次原始测试输入不对应。")
    if timing["model_sha256"] != sha256(args.model/"external_model/onnx/model_quantized.onnx"):
        raise ValueError("语义文件并非由包内固定网络生成。")
    if sha256(args.model/"channel.npz") != audit["final_channel_sha256"]:
        raise ValueError("通道参数与扩展审计后的交付配置不符。")
    semantic = dict(np.load(args.semantic_predictions))
    if not np.array_equal(frame.id.to_numpy(), semantic["id"]):
        raise ValueError("通道候选语义id错位。")
    old, base = inherited_from_frame(frame, args.model/"prior_model")
    sf = semantic_features(semantic["logits"], base)
    x = np.column_stack([old, base, sf]).astype(np.float32)
    p = predict_channel(x, semantic["logits"], dict(np.load(args.model/"channel.npz")))
    if not np.isfinite(p).all() or not ((p >= 1)&(p <= 5)).all():
        raise ValueError("通道评分非法。")
    pd.DataFrame({"id": frame.id, "rating": p}).to_csv(args.output, index=False, float_format="%.10f")
    write_json(args.output.with_suffix(".timing.json"), {"rows": len(frame),
        "post_semantic_feature_and_prediction_seconds": time.perf_counter()-began,
        "does_not_include_precomputed_semantic_inference": True, "sha256": sha256(args.output),
        "semantic_predictions_sha256": sha256(args.semantic_predictions),
        "candidate": "channel alpha1, selected in extended experiment after original frozen confirmation"})
    print("扩展通道候选完成", str(args.output), flush=True)


if __name__ == "__main__":
    main()
