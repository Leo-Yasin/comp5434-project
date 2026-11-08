"""对大体积评论 CSV 进行分块审计，不把全部文本载入内存。

主入口：D:/ProgramData/Anaconda3/python.exe -m task2.audit
保存只含哈希、标签、时间的紧凑元数据，供固定划分和重复记录分析使用。
这里的 64 位哈希用于审计和分组，理论上存在极小的碰撞概率。
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["id", "title", "user_id", "comment", "prod_id", "parent_prod_id", "time", "votes", "purchased"]
GROUP_FIELDS = ["title", "comment", "user_id", "prod_id", "parent_prod_id", "time"]
STRING_FIELDS = ["title", "user_id", "comment", "prod_id", "parent_prod_id", "purchased"]

# 显式固定 pandas 1.4 的空值词表：新版 pandas 会额外把字面量 "None" 当空值。
# 本项目将其保留为普通文本，保证不同 pandas 版本生成相同评论分组与文本特征。
CSV_OPTIONS = {"keep_default_na": False, "na_values": ["", "#N/A", "#N/A N/A", "#NA",
               "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>",
               "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]}


def write_json(path, value):
    """统一 UTF-8 中文记录，不允许悄悄写入非标准 NaN。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    """固定大小读取，避免为计算哈希额外分配 2 GB 内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(frame):
    """只统一缺失字符串；数值异常在审计中统计，不在这里删除记录。"""
    frame = frame.copy()
    for col in STRING_FIELDS:
        frame[col] = frame[col].fillna("").astype(str)
    return frame


def group_hash(frame):
    """相同评论身份与文本进入相同组，标签不参与分组或划分。"""
    return pd.util.hash_pandas_object(frame[GROUP_FIELDS], index=False).to_numpy(dtype=np.uint64)


def fixed_validation(groups, percent=2):
    """以独立于标签的稳定分组哈希划分；完全相同的评论不会跨训练/验证。"""
    return groups % np.uint64(100) < percent


def main():
    parser = argparse.ArgumentParser(description="Task 2 分块数据审计")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "comp-5434-20261-project-task-2")
    parser.add_argument("--output", type=Path, default=ROOT / "task2" / "audit")
    parser.add_argument("--chunk-size", type=int, default=100000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    test = normalize(pd.read_csv(args.data_dir / "test.csv", **CSV_OPTIONS))
    sample = pd.read_csv(args.data_dir / "sampleSubmission.csv", **CSV_OPTIONS)
    if list(sample.columns) != ["id", "rating"] or not test.id.equals(sample.id):
        raise ValueError("测试集与提交样例字段或顺序不一致。")
    if set(test.columns) != set(FIELDS) or not test.id.is_unique:
        raise ValueError("测试集字段或 id 不合法。")
    test_keys = {key: set(test[key]) for key in ["user_id", "prod_id", "parent_prod_id"]}
    test_groups = set(group_hash(test).tolist())
    seen_test = {key: set() for key in test_keys}
    train_groups, ratings, timestamps = [], [], []
    identifiers = {key: [] for key in test_keys}
    missing, labels = Counter(), Counter()
    rows, sequential, overlap = 0, True, 0
    total_text, max_text, empty_text = 0, 0, 0
    purchased = Counter()
    for number, chunk in enumerate(pd.read_csv(args.data_dir / "train.csv", chunksize=args.chunk_size, **CSV_OPTIONS)):
        if set(chunk.columns) != set(FIELDS + ["rating"]):
            raise ValueError("训练集字段不符。")
        missing.update({key: int(count) for key, count in chunk.isna().sum().items()})
        frame = normalize(chunk)
        y = pd.to_numeric(frame.rating, errors="raise").to_numpy()
        if not np.isin(y, [1, 2, 3, 4, 5]).all():
            raise ValueError("评分必须为 1–5。")
        labels.update({str(int(k)): int(v) for k, v in frame.rating.value_counts().items()})
        sequential &= np.array_equal(frame.id.to_numpy(), np.arange(rows, rows + len(frame)))
        groups = group_hash(frame)
        train_groups.append(groups)
        ratings.append(y.astype(np.int8))
        timestamps.append(pd.to_numeric(frame.time, errors="raise").to_numpy(dtype=np.int64))
        overlap += int(np.isin(groups, list(test_groups)).sum())
        for key in test_keys:
            hashes = pd.util.hash_pandas_object(frame[key], index=False).to_numpy(dtype=np.uint64)
            identifiers[key].append(hashes)
            seen_test[key].update(set(frame[key].unique()) & test_keys[key])
        lengths = frame.comment.str.len()
        total_text += int(lengths.sum())
        max_text = max(max_text, int(lengths.max()))
        empty_text += int((lengths == 0).sum())
        purchased.update({str(k): int(v) for k, v in frame.purchased.value_counts().items()})
        rows += len(frame)
        if number % 10 == 0:
            print(f"审计进度：{rows:,} 条，耗时 {time.perf_counter()-started:.1f}s", flush=True)
    groups = np.concatenate(train_groups)
    y = np.concatenate(ratings)
    times = np.concatenate(timestamps)
    validation = fixed_validation(groups)
    np.savez_compressed(args.output / "metadata.npz", groups=groups, rating=y, time=times)
    audit = {
        "train_rows": rows, "test_rows": len(test), "training_ids_are_sequential_from_zero": bool(sequential),
        "label_counts": dict(sorted(labels.items())), "missing_train": dict(missing),
        "test_missing": {key: int(v) for key, v in pd.read_csv(args.data_dir / "test.csv", **CSV_OPTIONS).isna().sum().items()},
        "empty_comment_count": empty_text, "mean_comment_characters": total_text / rows,
        "max_comment_characters": max_text, "purchased_counts": dict(purchased),
        "unique_group_hashes": int(len(np.unique(groups))),
        "duplicate_review_groups_extra_rows": int(rows - len(np.unique(groups))),
        "train_test_matching_review_group_rows": overlap,
        "validation_percent": 2, "validation_rows": int(validation.sum()),
        "fit_label_counts": {str(i): int(((y == i) & ~validation).sum()) for i in range(1, 6)},
        "validation_label_counts": {str(i): int(((y == i) & validation).sum()) for i in range(1, 6)},
        "time_min_ms": int(times.min()), "time_max_ms": int(times.max()),
        "identifiers": {}, "input_sha256": {},
    }
    for key, parts in identifiers.items():
        audit["identifiers"][key] = {
            "unique_training_hashes": int(len(np.unique(np.concatenate(parts)))),
            "unique_test_values": len(test_keys[key]), "seen_unique_test_values": len(seen_test[key]),
            "test_row_seen_fraction": float(test[key].isin(seen_test[key]).mean()),
        }
    for name in ["train.csv", "test.csv", "sampleSubmission.csv"]:
        audit["input_sha256"][name] = sha256(args.data_dir / name)
    audit["audit_seconds"] = time.perf_counter() - started
    write_json(args.output / "data_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
