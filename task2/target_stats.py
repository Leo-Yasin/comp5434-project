"""用户、商品及父商品的平滑评分统计，训练特征严格交叉拟合。

对验证记录，只使用外层训练部分的标签。对模型训练记录，再按不含标签的
评论分组哈希分成五组，排除自身所在整组后计算统计，避免把自身评分当输入。
完整模型与开发模型分别生成特征；测试记录使用全量训练集统计。
"""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from .audit import ROOT, write_json
from .features import ID_FIELDS, META_DIM


SMOOTHING = {"user_id": 5.0, "prod_id": 20.0, "parent_prod_id": 20.0}


def smooth_columns(sums, counts, inverse, prior, strength):
    """返回中心化平滑均值、对数支持数和冷启动标记。

    固定减去 3 而非某条记录的标签。没有历史记录时回退到可用训练部分均值。
    """
    n = counts[inverse]
    mean = (sums[inverse] + strength * prior) / (n + strength)
    return np.column_stack([mean - 3.0, np.log1p(n) / 10, n == 0]).astype(np.float32)


def fit_cross_fitted(keys, labels, groups, fit_mask, strength):
    """通过排序唯一键与 bincount 实现统计，不建立数百万 Python 字典。

    返回全体记录的交叉拟合特征和可保存的推理统计状态。
    内层先验也排除当前折，避免平滑先验间接包含自身评分。
    """
    unique, inverse = np.unique(keys, return_inverse=True)
    size = len(unique)
    sums = np.bincount(inverse[fit_mask], weights=labels[fit_mask], minlength=size)
    counts = np.bincount(inverse[fit_mask], minlength=size).astype(np.int32)
    prior = float(labels[fit_mask].mean())
    encoded = smooth_columns(sums, counts, inverse, prior, strength)
    folds = groups % np.uint64(5)
    for fold in range(5):
        held = fit_mask & (folds == fold)
        removed_sum = np.bincount(inverse[held], weights=labels[held], minlength=size)
        removed_count = np.bincount(inverse[held], minlength=size).astype(np.int32)
        kept_prior = float((labels[fit_mask].sum() - labels[held].sum()) / (fit_mask.sum() - held.sum()))
        encoded[held] = smooth_columns(sums - removed_sum, counts - removed_count,
                                       inverse[held], kept_prior, strength)
    return encoded, {"keys": unique, "sums": sums, "counts": counts,
                     "prior": np.array(prior), "strength": np.array(strength)}


def transform_keys(keys, state):
    """对任意新标识符做二分查找，未知标识符必须返回先验和冷启动标记。"""
    sorted_keys = state["keys"]
    indices = np.searchsorted(sorted_keys, keys)
    safe = np.minimum(indices, len(sorted_keys) - 1)
    found = (indices < len(sorted_keys)) & (sorted_keys[safe] == keys)
    count = np.where(found, state["counts"][safe], 0)
    total = np.where(found, state["sums"][safe], 0.0)
    prior, strength = float(state["prior"]), float(state["strength"])
    mean = (total + strength * prior) / (count + strength)
    return np.column_stack([mean - 3.0, np.log1p(count) / 10, count == 0]).astype(np.float32)


def prepare_statistics(cache):
    """只读取紧凑元数据，每次处理一个标识符字段；通过 memmap 控制内存。

    文本稀疏缓存不读取标签，评分统计另存并标明 split/full，防止混用。
    """
    cache = Path(cache)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    parts = manifest["parts"]
    groups, ratings = [], []
    for part in parts:
        with np.load(part["prefix"] + ".meta.npz") as data:
            groups.append(data["groups"])
            ratings.append(data["rating"])
    groups = np.concatenate(groups)
    labels = np.concatenate(ratings).astype(np.float64)
    outer_fit = groups % np.uint64(100) >= 2
    stats_dir = cache / "stats"
    stats_dir.mkdir(exist_ok=False)
    info = {"strengths": SMOOTHING, "inner_folds": 5, "modes": {}}
    test = dict(np.load(cache / "test.meta.npz"))
    for mode in ["split", "full"]:
        started = time.perf_counter()
        fit_mask = outer_fit if mode == "split" else np.ones(len(labels), dtype=bool)
        # 原始特征写入磁盘映射，避免将两份 768 万行数值矩阵都常驻内存。
        mapped = np.lib.format.open_memmap(stats_dir / f"{mode}.working.npy", mode="w+",
                                           dtype=np.float32, shape=(len(labels), META_DIM))
        test_columns = []
        for column, key in enumerate(ID_FIELDS):
            keys = []
            for part in parts:
                with np.load(part["prefix"] + ".meta.npz") as data:
                    keys.append(data[key])
            keys = np.concatenate(keys)
            encoded, state = fit_cross_fitted(keys, labels, groups, fit_mask, SMOOTHING[key])
            mapped[:, 3*column:3*column+3] = encoded
            mapped.flush()
            np.savez_compressed(stats_dir / f"{mode}.{key}.npz", **state)
            test_columns.append(transform_keys(test[key], state))
            print(f"目标统计 {mode}/{key} 已完成", flush=True)
            del keys, encoded, state
            gc.collect()
        for part in parts:
            first, last = part["start"], part["start"] + part["rows"]
            np.save(part["prefix"] + f".{mode}.numeric.npy", mapped[first:last])
        del mapped
        # 删除前已确认文件是当前 cache/stats 目录内明确定义的单个中间文件。
        working = (stats_dir / f"{mode}.working.npy").resolve()
        if working.parent != stats_dir.resolve():
            raise RuntimeError("中间文件路径越界。")
        working.unlink()
        np.save(cache / f"test.{mode}.numeric.npy", np.concatenate(test_columns, axis=1))
        info["modes"][mode] = {"fit_rows": int(fit_mask.sum()), "mean": float(labels[fit_mask].mean()),
                               "validation_rows": int((~fit_mask).sum()),
                               "seconds": time.perf_counter() - started}
    write_json(stats_dir / "manifest.json", info)
    return info


def main():
    parser = argparse.ArgumentParser(description="交叉拟合用户和商品统计")
    parser.add_argument("--cache", type=Path, default=ROOT / "task2" / "cache")
    args = parser.parse_args()
    prepare_statistics(args.cache)


if __name__ == "__main__":
    main()
