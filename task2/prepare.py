"""启动真正的 Dask 多进程集群，构建全量特征缓存。

本机执行：D:/ProgramData/Anaconda3/python.exe -m task2.prepare
"""

import argparse
from pathlib import Path

from distributed import Client, LocalCluster

from .audit import ROOT, write_json
from .features import build_cache


def main():
    parser = argparse.ArgumentParser(description="Dask 并行特征准备")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "comp-5434-20261-project-task-2")
    parser.add_argument("--output", type=Path, default=ROOT / "task2" / "cache")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50000)
    args = parser.parse_args()
    # 显式单线程工作进程，确保并行度代表真实进程数，并控制笔记本内存开销。
    with LocalCluster(n_workers=args.workers, threads_per_worker=1, processes=True,
                      dashboard_address=None, memory_limit="1GB", silence_logs=40,
                      local_directory=str(ROOT / "task2" / "tmp")) as cluster:
        with Client(cluster) as client:
            client.wait_for_workers(args.workers)
            result = build_cache(client, args.data_dir, args.output, args.chunk_size, args.workers)
            write_json(args.output / "cluster.json", {
                "framework": "Dask distributed", "workers": args.workers, "threads_per_worker": 1,
                "processes": True, "observed_task_worker_pids": result["worker_pids"],
            })
    print(f"缓存完成：{args.output}")


if __name__ == "__main__":
    main()
