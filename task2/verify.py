"""核对全部分组、开发 RMSE、独立模型包和重新预测文件。

该检查不伪造 Kaggle 分数，也不将当前验证集说成独立测试集。
大体积训练不重复整轮运行；重点验证数据对齐、优化数值不变量及独立推理。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .audit import ROOT, sha256, write_json
from .features import ID_FIELDS


def require(condition, message):
    """即使 Python -O，也必须保留这些检查。"""
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description="Task 2 完整产物核对")
    parser.add_argument("--cache", type=Path, default=ROOT / "task2/cache")
    parser.add_argument("--audit", type=Path, default=ROOT / "task2/audit_compatible")
    parser.add_argument("--development", type=Path, default=ROOT / "task2/outputs/development")
    parser.add_argument("--final", type=Path, default=ROOT / "task2/outputs/final")
    parser.add_argument("--reproduced", type=Path, default=ROOT / "task2/outputs/reproduced_submission.csv")
    args = parser.parse_args()
    cache_info = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((args.audit / "data_audit.json").read_text(encoding="utf-8"))
    raw_metadata = np.load(args.audit / "metadata.npz")
    groups, labels = raw_metadata["groups"], raw_metadata["rating"]
    mask = groups % np.uint64(100) < 2
    checked = 0
    for part in cache_info["parts"]:
        start, end = part["start"], part["start"] + part["rows"]
        with np.load(part["prefix"] + ".meta.npz") as data:
            require(np.array_equal(data["groups"], groups[start:end]), "缓存与兼容审计的分组哈希不一致。")
            require(np.array_equal(data["rating"], labels[start:end]), "缓存与审计评分不一致。")
            require(np.array_equal(data["id"], np.arange(start, end)), "缓存训练记录位置错位。")
        checked += part["rows"]
    require(checked == len(groups) == audit["train_rows"], "缓存遗漏训练记录。")
    stats_info = json.loads((args.cache / "stats/manifest.json").read_text(encoding="utf-8"))
    require(int(mask.sum()) == stats_info["modes"]["split"]["validation_rows"], "统计与验证行数不一致。")
    comparison = json.loads((args.development / "comparison.json").read_text(encoding="utf-8"))
    result = {"status": "passed", "all_training_rows_checked": checked,
              "group_hashes_match_compatible_audit": True, "validation_rows": int(mask.sum()),
              "models": {}, "full_training_repeated": False}
    for summary in comparison:
        frame = pd.read_csv(args.development / summary["variant"] / "validation_predictions.csv")
        require(np.array_equal(frame.id.to_numpy(), np.flatnonzero(mask)), "验证预测顺序错位。")
        require(np.array_equal(frame.rating.to_numpy(), labels[mask]), "验证真实评分错位。")
        error = (frame.rating - frame.prediction) ** 2
        score = float(np.sqrt(error.mean()))
        require(abs(score - summary["validation_metrics"]["rmse"]) < 1e-10, "验证 RMSE 无法重算。")
        result["models"][summary["variant"]] = {"rmse_recalculated": score}
    baseline = float(np.sqrt(np.mean((labels[mask] - stats_info["modes"]["split"]["mean"]) ** 2)))
    result["global_mean_validation_rmse"] = baseline
    model = args.final / "model"
    description = json.loads((model / "model.json").read_text(encoding="utf-8"))
    for name, expected in description["files_sha256"].items():
        require(sha256(model / name) == expected, "模型包文件哈希不匹配。")
    require(description["training_csv_sha256"] == audit["input_sha256"]["train.csv"], "最终模型训练数据哈希不匹配。")
    submission = pd.read_csv(args.final / "submission.csv")
    sample = pd.read_csv(ROOT / "comp-5434-20261-project-task-2/sampleSubmission.csv")
    require(list(submission.columns) == ["id", "rating"], "提交字段不正确。")
    require(submission.id.equals(sample.id), "提交 id 与样例不一致。")
    require(np.isfinite(submission.rating).all() and submission.rating.between(1, 5).all(), "预测评分不合法。")
    require((args.final / "submission.csv").read_bytes() == args.reproduced.read_bytes(), "独立 CLI 预测文件不一致。")
    result["independent_cli_prediction_byte_identical"] = True
    result["submission_rows"] = len(submission)
    result["submission_sha256"] = sha256(args.final / "submission.csv")

    # 使用训练缓存中的测试特征再计算一遍，核对独立模型包与缓存数据流一致。
    state = np.load(model / "weights.npz")
    variant = description["variant"]
    numeric = np.load(args.cache / "test.full.numeric.npy")
    if variant == "metadata":
        x = sparse.csr_matrix(np.column_stack([np.ones(len(numeric)), numeric]), dtype=np.float32)
    else:
        x = sparse.load_npz(args.cache / "test.text.npz")
        if variant == "hybrid":
            x = sparse.hstack([x, sparse.csr_matrix(numeric)], format="csr", dtype=np.float32)
    expected = np.clip(float(state["mean"]) + x @ state["weights"], 1, 5)
    require(np.allclose(expected, submission.rating, rtol=0, atol=5.1e-11), "独立推理与缓存特征流不同。")
    result["cached_feature_and_raw_csv_inference_match"] = True
    write_json(args.final / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
