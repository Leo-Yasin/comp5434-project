"""将最终全量模型打包成不依赖训练缓存的目录，并生成第一次提交预测。"""

import argparse
import json
from pathlib import Path
import shutil

from .audit import ROOT, sha256, write_json
from .features import ID_FIELDS
from .predict import predict_file


def main():
    parser = argparse.ArgumentParser(description="导出 Task 2 独立模型包")
    parser.add_argument("--run", type=Path, default=ROOT / "task2/outputs/final")
    parser.add_argument("--cache", type=Path, default=ROOT / "task2/cache")
    parser.add_argument("--variant", choices=["metadata", "text", "hybrid"], default="hybrid")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "comp-5434-20261-project-task-2")
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    if config["mode"] != "full":
        raise ValueError("只能将最终全量训练的模型导出为提交候选。")
    model = args.run / "model"
    model.mkdir(exist_ok=False)
    shutil.copy2(args.run / args.variant / "last_weights.npz", model / "weights.npz")
    if args.variant in ["metadata", "hybrid"]:
        for key in ID_FIELDS:
            shutil.copy2(args.cache / "stats" / f"full.{key}.npz", model / f"{key}.npz")
    description = {"schema_version": 1, "algorithm": "custom ridge regression with Dask PCG",
                   "variant": args.variant, "configuration": config,
                   "training_csv_sha256": sha256(args.data_dir / "train.csv"),
                   "files_sha256": {path.name: sha256(path) for path in model.iterdir() if path.is_file()}}
    write_json(model / "model.json", description)
    predict_file(model, args.data_dir / "test.csv", args.run / "submission.csv")


if __name__ == "__main__":
    main()
