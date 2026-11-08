"""从独立模型包读取统计和权重，为所有测试记录预测连续评分。

不读取训练评分或特征缓存。计时包含模型加载、CSV 读取、特征构建和推理，
不包含 CSV 写盘，便于报告完整的测试阶段计算开销。
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse

from .audit import CSV_OPTIONS, normalize, sha256, write_json
from .features import ID_FIELDS, build_text
from .target_stats import transform_keys


def predict_file(model_dir, test_path, output):
    """输出列严格为 id、rating，保持原始行顺序，预测限制在 [1,5]。"""
    model_dir, test_path, output = Path(model_dir), Path(test_path), Path(output)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有提交文件：{output}")
    started = time.perf_counter()
    config = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    if config["schema_version"] != 1:
        raise ValueError("不支持的模型格式。")
    state = np.load(model_dir / "weights.npz")
    weights, mean = state["weights"], float(state["mean"])
    frame = normalize(pd.read_csv(test_path, **CSV_OPTIONS))
    if not frame.id.is_unique or frame.id.isna().any():
        raise ValueError("测试记录 id 缺失或重复。")
    variant = config["variant"]
    if variant in ["metadata", "hybrid"]:
        columns = []
        for key in ID_FIELDS:
            with np.load(model_dir / f"{key}.npz") as statistics:
                hashes = pd.util.hash_pandas_object(frame[key], index=False).to_numpy(dtype=np.uint64)
                columns.append(transform_keys(hashes, statistics))
        numeric = np.concatenate(columns, axis=1)
    if variant == "metadata":
        x = sparse.csr_matrix(np.column_stack([np.ones(len(frame)), numeric]), dtype=np.float32)
    else:
        x = build_text(frame)
        if variant == "hybrid":
            x = sparse.hstack([x, sparse.csr_matrix(numeric)], format="csr", dtype=np.float32)
    prediction = np.clip(mean + x @ weights, 1.0, 5.0)
    if config.get("score_calibration") is not None:
        # 只在模型包显式记录该假设与训练估计参数时校正；旧模型行为完全保留。
        from .calibration import correct
        prediction=correct(prediction,config["score_calibration"]["epsilon"])
    seconds = time.perf_counter() - started
    if not np.isfinite(prediction).all():
        raise ValueError("预测含非有限值。")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": frame.id.to_numpy(), "rating": prediction}).to_csv(
        output, index=False, float_format="%.10f")
    timing = {"records": len(frame), "inference_seconds_including_loading_reading_transform": seconds,
              "minimum_rating": float(prediction.min()), "maximum_rating": float(prediction.max()),
              "mean_rating": float(prediction.mean()), "submission_sha256": sha256(output)}
    write_json(output.with_suffix(".timing.json"), timing)
    print(f"已生成 {len(frame):,} 条评分，完整推理耗时 {seconds:.3f}s：{output}", flush=True)
    return timing


def main():
    parser = argparse.ArgumentParser(description="独立复现 Task 2 评分预测")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predict_file(args.model, args.test, args.output)


if __name__ == "__main__":
    main()
