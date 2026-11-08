"""比较相同训练核心算子在 1 和 3 个 Dask 工作进程上的实际开销。

这不是全量端到端训练速度基准。固定前 12 块（最多 60 万条）、同一方向向量、
同一 split/hybrid 数据和重复次数。每任务清除 Python LRU，避免进程数增加
同时扩大 LRU 容量成为主要混淆因素；操作系统文件缓存无法完全控制，如实说明。
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from distributed import Client, LocalCluster

from .audit import ROOT, write_json
from . import features
from .features import DIM
from .train import partition_matvec


def kernel(prefix, direction):
    """每个任务都从相同的内存映射缓存重新建立矩阵视图。"""
    features._CACHE.clear()
    result = partition_matvec(prefix, "split", "hybrid", direction)
    features._CACHE.clear()
    return result


def main():
    parser = argparse.ArgumentParser(description="训练核心算子并行效率比较")
    parser.add_argument("--cache", type=Path, default=ROOT / "task2/cache")
    parser.add_argument("--output", type=Path, default=ROOT / "task2/outputs/benchmark.json")
    parser.add_argument("--parts", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    parts = manifest["parts"][:args.parts]
    rng = np.random.default_rng(5434)
    direction = rng.normal(scale=0.01, size=DIM)
    result = {"kernel": "X_fit.T @ (X_fit @ direction)", "rows": sum(p["rows"] for p in parts),
              "parts": len(parts), "repeats": args.repeats, "cache": "memory-mapped CSR",
              "timing_includes": "task submission, cache opening, matrix multiplication, ordered reduction",
              "timing_excludes": "cluster startup and direction broadcast",
              "limitation": "OS 文件缓存和系统背景负载未完全控制；不是全量端到端训练加速比。",
              "configurations": []}
    reference = None
    for workers in [1, 3]:
        times = []
        with LocalCluster(n_workers=workers, threads_per_worker=1, processes=True,
                          dashboard_address=None, memory_limit="1GB", silence_logs=40,
                          local_directory=str(ROOT / "task2/tmp")) as cluster:
            with Client(cluster) as client:
                client.wait_for_workers(workers)
                vector = client.scatter([direction], broadcast=True, hash=False)[0]
                for repeat in range(args.repeats):
                    started = time.perf_counter()
                    futures = [client.submit(kernel, part["prefix"], vector, pure=False) for part in parts]
                    total = np.zeros(DIM)
                    for future in futures:
                        total += future.result()
                        future.release()
                    times.append(time.perf_counter() - started)
                    if reference is None:
                        reference = total.copy()
                    elif not np.allclose(reference, total, rtol=1e-10, atol=1e-8):
                        raise RuntimeError("并行配置改变后算子结果不同。")
                    print(f"算子 benchmark workers={workers}, repeat={repeat+1}: {times[-1]:.3f}s", flush=True)
        result["configurations"].append({"workers": workers, "threads_per_worker": 1,
                                         "seconds": times, "median_seconds": float(np.median(times))})
    result["median_speedup_3_vs_1"] = result["configurations"][0]["median_seconds"] / result["configurations"][1]["median_seconds"]
    result["all_results_numerically_identical"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
