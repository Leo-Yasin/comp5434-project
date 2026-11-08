"""第五轮完整独立预测：原始测试 CSV + 自包含模型包 + 独立 ONNX 环境。

主进程使用原 Python 3.9 数值环境；子进程只负责冻结语义网络，不访问标签。
所有子进程参数用列表传递，路径含空格也可运行，不拼接 shell 命令。
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import ROOT, CSV_OPTIONS, sha256, write_json
from .features import TEXT_DIM, build_text
from .round3_features import features_from_frame
from .round4_features import refinement_matrix
from .lookup_model import residual_design
from .regression_boost import ResidualHistogramBoost
from .semantic_fusion import semantic_features, predict_linear, known_history_mask
from .noise_softmax import predict_channel
from .semantic_train import chosen_prediction


def inherited_from_frame(frame, model):
    """从包内参数重建第四轮39列和评分，不依赖任何训练缓存/旧提交预测。"""
    description = json.loads((model/"model.json").read_text(encoding="utf-8"))
    previous = model/description["prior"]
    previous_description = json.loads((previous/"model.json").read_text(encoding="utf-8"))
    support = previous/previous_description["support"]
    eps = json.loads((support/"model.json").read_text(encoding="utf-8"))["epsilon_full"]
    x, base, means, counts, context = features_from_frame(frame, support)
    coefficient = np.load(previous/"linear.npz")["coefficient"]
    initial = base+residual_design(base, means, counts, context)@coefficient
    old = ResidualHistogramBoost.from_dict(json.loads((previous/"booster.json").read_text(encoding="utf-8")))
    inherited = initial+previous_description["alpha"]*(old.predict(x, initial)-initial)
    auxiliary = np.zeros(len(frame))
    chosen = description["chosen"]
    if chosen["use_auxiliary"]:
        state = np.load(model/"auxiliary.npz")
        numeric = context.copy(); numeric[:, [0, 3, 6]] *= 1-eps
        text = build_text(frame)
        auxiliary = float(state["mean"])+text@state["weights"][:TEXT_DIM]+numeric.astype(np.float32)@state["weights"][TEXT_DIM:]
    features = refinement_matrix(x, inherited, auxiliary, eps, chosen["use_auxiliary"])
    booster = ResidualHistogramBoost.from_dict(json.loads((model/"booster.json").read_text(encoding="utf-8")))
    prediction = np.clip(inherited+chosen["alpha"]*(booster.predict(features, inherited)-inherited), 1, 5)
    return features, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--semantic-python", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.work_dir.resolve() == args.model.resolve():
        raise ValueError("工作目录不能是模型根目录。")
    began = time.perf_counter()
    description = json.loads((args.model/"model.json").read_text(encoding="utf-8"))
    if description["schema_version"] != 1:
        raise ValueError("未知语义模型包版本。")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.test, **CSV_OPTIONS)
    input_file = args.work_dir/"test_text.csv"
    frame[["id", "title", "comment"]].fillna("").to_csv(input_file, index=False)
    semantic_file = args.work_dir/"semantic.npz"
    runtime = description["semantic_runtime"]
    command = [str(args.semantic_python.resolve()), "-X", "utf8", "-m", "task2.semantic_infer",
        "--model", str((args.model/description["external_model"]).resolve()),
        "--input", str(input_file.resolve()), "--output", str(semantic_file.resolve()),
        "--provider", runtime["provider"], "--max-length", str(runtime["max_length"]),
        "--batch-size", str(runtime["batch_size"])]
    if runtime["dynamic_length"]:
        command.append("--dynamic-length")
    subprocess.run(command, check=True, cwd=ROOT)
    semantic = dict(np.load(semantic_file))
    if not np.array_equal(frame.id.to_numpy(), semantic["id"]):
        raise ValueError("独立语义预测id错位。")
    prior_features, prior = inherited_from_frame(frame, args.model/description["prior_model"])
    semantic_matrix = semantic_features(semantic["logits"], prior)
    features = np.column_stack([prior_features, prior, semantic_matrix]).astype(np.float32)
    linear = predict_linear(prior, semantic_matrix, dict(np.load(args.model/"linear.npz")))
    channel = predict_channel(features, semantic["logits"], dict(np.load(args.model/"channel.npz")))
    booster = ResidualHistogramBoost.from_dict(json.loads((args.model/"booster.json").read_text(encoding="utf-8")))
    predictions = {"baseline": prior, "linear": linear, "channel": channel}
    predictions["channel_warm"] = np.where(known_history_mask(prior_features),
        predict_channel(features, semantic["logits"], dict(np.load(args.model/"channel_warm.npz"))), channel)
    for rounds in [100, 200, 400]:
        predictions[f"tree{rounds}"] = booster.predict(features, linear, rounds)
    p = np.clip(chosen_prediction(predictions, description["chosen"]), 1, 5)
    if not np.isfinite(p).all() or not frame.id.is_unique:
        raise ValueError("最终评分或 id 不合法。")
    pd.DataFrame({"id": frame.id, "rating": p}).to_csv(args.output, index=False, float_format="%.10f")
    write_json(args.output.with_suffix(".timing.json"), {"rows": len(frame),
        "complete_inference_seconds": time.perf_counter()-began, "includes_semantic_and_all_features": True,
        "semantic_checkpoint_reused": json.loads(semantic_file.with_suffix(".timing.json").read_text())["rows_resumed"] > 0,
        "sha256": sha256(args.output), "external_model_rule": description["competition_external_model_rule"]})
    print("第五轮完整独立预测完成", str(args.output), flush=True)


if __name__ == "__main__":
    main()
