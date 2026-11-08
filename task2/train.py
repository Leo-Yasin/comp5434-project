"""Dask 分布式矩阵运算 + 自主实现预条件共轭梯度（PCG）评分回归。

目标：min_w sum_i (x_i w + mean - y_i)^2 + lambda ||w||²。
不构造稠密 X.T X；每次迭代由工作进程计算 X_k.T (X_k p)，主进程
按固定分区顺序求和并更新参数。稀疏原始特征始终留在工作进程或磁盘缓存。
"""

import argparse
import ctypes
from datetime import datetime, timezone, timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy import sparse
from distributed import Client, LocalCluster

from .audit import ROOT, sha256, write_json
from .features import (BODY_DIM, DIM, META_DIM, NUMERIC_DIM, TEXT_DIM, TITLE_DIM,
                       load_partition)


def hardware_info():
    """记录实际机器信息；读取不到时明确回退，不把架构字符串当成 CPU 型号。"""
    result = {"cpu_model": None, "logical_cpu_count": os.cpu_count(),
              "servers": 1, "gpu_used": False}
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                result["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            memory_type = type("MemoryStatus", (ctypes.Structure,), {"_fields_": [
                ("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (key, ctypes.c_ulonglong) for key in ["total", "available", "page_total",
                 "page_available", "virtual_total", "virtual_available", "extended"]]})
            memory = memory_type()
            memory.length = ctypes.sizeof(memory)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
            result["ram_bytes"] = int(memory.total)
        except OSError:
            pass
    return result


def partition_statistics(prefix, mode, variant, mean):
    """仅使用当前训练部分，计算正规方程右端项和 Jacobi 预条件器对角。"""
    x, y, fit, _, _ = load_partition(prefix, mode, variant)
    residual = (y - mean) * fit
    right = np.asarray(x.T @ residual).ravel()
    diagonal = np.asarray(x.multiply(x).T @ fit.astype(float)).ravel()
    return right, diagonal, os.getpid()


def partition_matvec(prefix, mode, variant, direction):
    """不把验证行用于优化；先计算 Xp，再把验证位置置零。"""
    x, _, fit, _, _ = load_partition(prefix, mode, variant)
    projection = x @ direction
    projection[~fit] = 0.0
    return np.asarray(x.T @ projection).ravel()


def partition_prediction(prefix, mode, variant, weights, mean):
    """开发阶段只返回外层验证记录；warm 指用户和商品均有外层训练历史。"""
    x, y, _, valid, ids = load_partition(prefix, mode, variant)
    prediction = np.clip(mean + x[valid] @ weights, 1.0, 5.0)
    numeric = np.load(prefix + ".split.numeric.npy", mmap_mode="r")
    warm = (numeric[valid, 2] == 0) & (numeric[valid, 5] == 0)
    return ids[valid], y[valid], prediction, warm


def reduce_vectors(client, parts, function, *arguments):
    """结果按固定分区顺序归约，避免完成顺序改变浮点累加结果。

    逐个取 Future 结果后立即释放，避免同时持有所有稠密梯度向量。
    """
    futures = [client.submit(function, part["prefix"], *arguments, pure=False) for part in parts]
    total = None
    for future in futures:
        value = future.result()
        if total is None:
            total = np.array(value, dtype=np.float64, copy=True)
        else:
            total += value
        future.release()
    return total


def evaluate(client, parts, variant, weights, mean):
    """RMSE 使用连续且截断到 [1,5] 的预测，不四舍五入成类别。"""
    weight_future = client.scatter([weights], broadcast=True, hash=False)[0]
    futures = [client.submit(partition_prediction, part["prefix"], "split", variant,
                             weight_future, mean, pure=False) for part in parts]
    records = [future.result() for future in futures]
    for future in futures:
        future.release()
    weight_future.release()
    ids, labels, predictions, warm = [np.concatenate([record[i] for record in records]) for i in range(4)]
    errors = (labels - predictions) ** 2
    metrics = {"rmse": float(np.sqrt(errors.mean())), "rows": len(labels),
               "warm_rows": int(warm.sum()),
               "warm_rmse": float(np.sqrt(errors[warm].mean())) if warm.any() else None,
               "cold_rmse": float(np.sqrt(errors[~warm].mean())) if (~warm).any() else None}
    return metrics, (ids, labels, predictions, warm)


def pcg_fit(client, parts, mode, variant, mean, ridge, epochs, output, evaluate_every=5, initial_weights=None):
    """求解 (X.T X + R)w = X.T(y-mean)，采用对角预条件改善收敛。

    一次 PCG 迭代记为一次训练迭代；不是神经网络的 mini-batch epoch。
    每次迭代会对全部训练分区做一次矩阵乘法，报告按完整遍历计时。
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    futures = [client.submit(partition_statistics, p["prefix"], mode, variant, mean, pure=False)
               for p in parts]
    right = diagonal = None
    worker_pids = set()
    for future in futures:
        rhs, diag, pid = future.result()
        if right is None:
            right, diagonal = rhs.copy(), diag.copy()
        else:
            right += rhs
            diagonal += diag
        worker_pids.add(pid)
        future.release()
    penalty = np.full(len(right), ridge)
    # 截距几乎不惩罚；其他特征使用同一 ridge，定义明确且可复现。
    intercept = 0 if variant == "metadata" else TITLE_DIM + BODY_DIM
    penalty[intercept] = 1e-6
    diagonal += penalty
    weights = np.zeros(len(right), dtype=np.float64)
    residual = right.copy()
    if initial_weights is not None:
        # 延续旧模型的权重，但重新计算正规方程残差并重启搜索方向。
        # 这不是复用旧 PCG 方向，因此日志称为 warm-start 新迭代，而非等同冷启动轮数。
        weights = np.asarray(initial_weights,dtype=np.float64).copy()
        if weights.shape != right.shape:
            raise ValueError("初始权重维度不匹配。")
        old_future = client.scatter([weights],broadcast=True,hash=False)[0]
        residual -= reduce_vectors(client,parts,partition_matvec,mode,variant,old_future)+penalty*weights
        old_future.release()
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    rz = float(residual @ preconditioned)
    initial_norm = max(float(np.linalg.norm(right)), 1e-30)
    history, best = [], None
    setup_seconds = time.perf_counter() - started
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        direction_future = client.scatter([direction], broadcast=True, hash=False)[0]
        product = reduce_vectors(client, parts, partition_matvec, mode, variant, direction_future)
        direction_future.release()
        product += penalty * direction
        denominator = float(direction @ product)
        if denominator <= 0:
            raise RuntimeError("PCG 正定性检查失败。")
        step = rz / denominator
        weights += step * direction
        residual -= step * product
        new_preconditioned = residual / diagonal
        next_rz = float(residual @ new_preconditioned)
        direction = new_preconditioned + (next_rz / max(rz, 1e-300)) * direction
        preconditioned, rz = new_preconditioned, next_rz
        relative_residual = float(np.linalg.norm(residual) / initial_norm)
        row = {"iteration": epoch, "training_seconds": time.perf_counter() - epoch_start,
               "relative_normal_equation_residual": relative_residual}
        if mode == "split" and (epoch % evaluate_every == 0 or epoch == epochs or relative_residual < 1e-6):
            evaluation_start = time.perf_counter()
            metrics, _ = evaluate(client, parts, variant, weights, mean)
            row["validation"] = metrics
            row["validation_seconds"] = time.perf_counter() - evaluation_start
            if best is None or metrics["rmse"] < best["rmse"]:
                best = {"rmse": metrics["rmse"], "iteration": epoch}
                np.savez_compressed(output / "best_weights.npz", weights=weights, mean=mean)
        history.append(row)
        write_json(output / "history.json", history)
        score = f", validation RMSE={row['validation']['rmse']:.6f}" if "validation" in row else ""
        print(f"{mode}/{variant} iteration {epoch}/{epochs}: {row['training_seconds']:.2f}s{score}", flush=True)
        if relative_residual < 1e-6:
            break
    training_seconds = time.perf_counter() - started
    np.savez_compressed(output / "last_weights.npz", weights=weights, mean=mean)
    result = {"mode": mode, "variant": variant, "ridge": ridge, "mean": mean,
              "dimensions": len(weights), "iterations": len(history),
              "setup_statistics_seconds": setup_seconds,
              "total_fit_seconds_including_validation": training_seconds,
              "training_iteration_seconds": [row["training_seconds"] for row in history],
              "mean_training_iteration_seconds": float(np.mean([row["training_seconds"] for row in history])),
              "observed_statistics_worker_pids": sorted(worker_pids), "best": best}
    if mode == "split":
        chosen = np.load(output / "best_weights.npz")
        metrics, records = evaluate(client, parts, variant, chosen["weights"], float(chosen["mean"]))
        ids, labels, predictions, warm = records
        import pandas as pd
        pd.DataFrame({"id": ids, "rating": labels, "prediction": predictions, "warm": warm}).to_csv(
            output / "validation_predictions.csv", index=False)
        result["validation_metrics"] = metrics
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Dask 并行自定义 Ridge-PCG 评分模型")
    parser.add_argument("--cache", type=Path, default=ROOT / "task2" / "cache")
    parser.add_argument("--output", type=Path, default=ROOT / "task2" / "outputs" / "development")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=20.0)
    parser.add_argument("--variants", nargs="+", choices=["metadata", "text", "hybrid"],
                        default=["metadata", "text", "hybrid"])
    parser.add_argument("--mode", choices=["split", "full"], default="split")
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--evaluate-every",type=int,default=5)
    args = parser.parse_args()
    if args.ridge <= 0 or args.epochs < 1 or args.workers < 1:
        parser.error("ridge、迭代次数和进程数必须为正。")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output / "config.json", config)
    cache_info = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    stats_info = json.loads((args.cache / "stats/manifest.json").read_text(encoding="utf-8"))
    environment = {"python": sys.version, "platform": platform.platform(), **hardware_info(),
                   "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                   "workers": args.workers, "threads_per_worker": 1,
                   "packages": {name: importlib.metadata.version(name) for name in
                                ["numpy", "pandas", "scipy", "scikit-learn", "dask", "distributed"]},
                   "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                   "framework": "Dask distributed.LocalCluster", "status": "running"}
    write_json(args.output / "manifest.json", environment)
    try:
        cluster_start = time.perf_counter()
        with LocalCluster(n_workers=args.workers, threads_per_worker=1, processes=True,
                          dashboard_address=None, memory_limit="1GB", silence_logs=40,
                          local_directory=str(ROOT / "task2" / "tmp")) as cluster:
            with Client(cluster) as client:
                client.wait_for_workers(args.workers)
                environment["cluster_setup_seconds"] = time.perf_counter() - cluster_start
                mean = stats_info["modes"][args.mode]["mean"]
                comparison = []
                for variant in args.variants:
                    result = pcg_fit(client, cache_info["parts"], args.mode, variant, mean,
                                     args.ridge, args.epochs, args.output / variant,args.evaluate_every,
                                     np.load(args.initial_weights)["weights"] if args.initial_weights else None)
                    comparison.append(result)
                    write_json(args.output / "comparison.json", comparison)
        environment["status"] = "completed"
        environment["finished_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        write_json(args.output / "manifest.json", environment)
    except Exception as error:
        environment.update(status="failed", error=repr(error))
        write_json(args.output / "manifest.json", environment)
        raise


if __name__ == "__main__":
    main()
