"""依据已保存开发结果选择方案/轮数，训练全量数据并导出独立模型包。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .audit import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description="按开发结果完成全量模型及提交预测")
    parser.add_argument("--development", type=Path, default=ROOT / "task2/outputs/development")
    parser.add_argument("--cache", type=Path, default=ROOT / "task2/cache")
    parser.add_argument("--output", type=Path, default=ROOT / "task2/outputs/final")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "comp-5434-20261-project-task-2")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads((args.development / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("开发实验尚未全部完成，不能选择最终方案。")
    comparison = json.loads((args.development / "comparison.json").read_text(encoding="utf-8"))
    best = min(comparison, key=lambda row: row["validation_metrics"]["rmse"])
    variant, epochs, ridge = best["variant"], best["best"]["iteration"], best["ridge"]
    print(f"选择 {variant}，开发 RMSE={best['validation_metrics']['rmse']:.6f}，全量迭代 {epochs} 次", flush=True)
    # 使用同一解释器，不经 shell 拼接，正确处理带空格的项目目录。
    subprocess.run([sys.executable, "-X", "utf8", "-m", "task2.train", "--mode", "full",
                    "--variants", variant, "--epochs", str(epochs), "--ridge", str(ridge),
                    "--workers", str(args.workers), "--cache", str(args.cache),
                    "--output", str(args.output)], cwd=ROOT, check=True)
    write_json(args.output / "selection.json", {
        "criterion": "lowest development validation RMSE among saved variants/checkpoints",
        "selected_variant": variant, "selected_iterations": epochs,
        "development_validation_rmse": best["validation_metrics"]["rmse"],
        "independent_test_score": None,
        "candidates": [{"variant": row["variant"], "rmse": row["validation_metrics"]["rmse"],
                         "iteration": row["best"]["iteration"]} for row in comparison],
    })
    subprocess.run([sys.executable, "-X", "utf8", "-m", "task2.export", "--run", str(args.output),
                    "--cache", str(args.cache), "--variant", variant,
                    "--data-dir", str(args.data_dir)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
