"""将压缩 CSR 转成三个可按需映射的数组，降低多次训练遍历的解压成本。

原压缩缓存保留，便于核对和迁移；额外磁盘开销约 3.6 GiB。
转换必须逐个核对三个数组相同，不能因追求速度改变任何特征值。
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from .audit import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description="CSR 缓存磁盘映射优化")
    parser.add_argument("--cache", type=Path, default=ROOT / "task2/cache")
    args = parser.parse_args()
    manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    total_bytes = 0
    for part in manifest["parts"]:
        prefix = part["prefix"]
        matrix = sparse.load_npz(prefix + ".text.npz")
        for name in ["data", "indices", "indptr"]:
            path = Path(prefix + f".{name}.npy")
            if path.exists():
                raise FileExistsError(f"拒绝覆盖已有映射缓存：{path}")
            array = getattr(matrix, name)
            np.save(path, array)
            mapped = np.load(path, mmap_mode="r")
            if not np.array_equal(array, mapped):
                raise RuntimeError("缓存转换后数组不相同。")
            total_bytes += path.stat().st_size
        if part["number"] % 20 == 0:
            print(f"映射缓存已转换 {part['number']+1} 块", flush=True)
    write_json(args.cache / "mapping_optimization.json", {
        "conversion_seconds": time.perf_counter() - started, "bytes": total_bytes,
        "all_arrays_identical": True, "parts": len(manifest["parts"]),
    })


if __name__ == "__main__":
    main()
