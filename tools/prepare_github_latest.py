"""整理GitHub最新版快照；不修改原训练产物，不执行远程推送。

仅复制明确允许的源码/说明文件，两个任务各保留一个最新CSV。
Task1模型足够小，完整复制；Task2大型模型单独共享，不能伪装为可直接预测的完整包。
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"tmp/github-publish-20261108"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def copy(source, relative):
    target = DEST/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    if not (DEST/".git").exists():
        raise RuntimeError("目标必须是事先克隆的指定仓库。")
    # 保留当前源码及其基础模块，避免只复制入口脚本导致导入失败。
    for task in ["task1", "task2"]:
        for file in (ROOT/task).iterdir():
            if file.is_file() and (file.suffix in [".py", ".md"] or file.name.startswith("requirements") or file.name == ".gitignore"):
                copy(file, file.relative_to(ROOT))
        for folder in ["docs", "tests"]:
            for file in (ROOT/task/folder).rglob("*"):
                if file.is_file() and file.suffix in [".py", ".md"]:
                    copy(file, file.relative_to(ROOT))
    for file in (ROOT/"docs").glob("*.md"):
        copy(file, file.relative_to(ROOT))
    for file in (ROOT/"tools").glob("*.py"):
        copy(file, file.relative_to(ROOT))
    copy(ROOT/".gitignore", ".gitignore")
    selected = {
        "task1": ("submissions/20261105_round2/task1_experimental_mixture.csv",
                  "bd58fb555decd6debfe13a3b80ed13662d626b4e23ab2130771cae2b08f9f94b", 3300),
        "task2": ("submissions/20261105_task2_round5/task2_semantic_noise_fusion.csv",
                  "0be1e6c8b4d7fa2ccf16921cc97077f8f7c0c4adff409f5ef13dd20ea132a126", 10000)}
    manifest = {"snapshot_date": "2026-11-08", "only_latest_submission_per_task": True,
        "repository": "https://github.com/Leo-Yasin/comp5434-project", "tasks": {}}
    for task, (source, expected, rows) in selected.items():
        if digest(ROOT/source) != expected:
            raise RuntimeError("原提交哈希改变："+task)
        copy(ROOT/source, f"submissions/latest/{task}.csv")
        manifest["tasks"][task] = {"source_local_path": source, "rows": rows, "sha256": expected,
            "submitted_file": f"submissions/latest/{task}.csv"}
    # 三个JSON构成Task1完整推理包；不导出任何训练样本或测试真值。
    model = ROOT/"task1/outputs/round2_20261105/package"
    for file in model.rglob("*.json"):
        copy(file, Path("task1/latest/model")/file.relative_to(model))
    manifest["tasks"]["task1"].update({"method": "mixed Gaussian/categorical model and previous boosted-forest ensemble",
        "model": "task1/latest/model", "exact_latest_kaggle_score": None,
        "user_reported_score": "0.913级别，小幅提升，精确值未提供"})
    manifest["tasks"]["task2"].update({"method": "frozen BERT semantic features and custom nonlinear residual fusion",
        "model_included": False, "latest_kaggle_score": None,
        "previous_measured_best": .59347, "external_model_rules_verified": False})
    for source in ["frozen_selection.json", "confirmation.json", "extra_audit.json", "paired_uncertainty.json", "reproduced.timing.json"]:
        copy(ROOT/"task2/outputs/round5_20261105"/source, Path("task2/latest")/source)
    copy(ROOT/"submissions/20261105_task2_round5/manifest.json", "task2/latest/verification_manifest.json")
    copy(ROOT/"submissions/20261105_round2/manifest.json", "task1/latest/source_verification_manifest.json")
    manifest["tasks"]["task1"]["model_hashes"] = {p.relative_to(DEST/"task1/latest/model").as_posix(): digest(p)
        for p in (DEST/"task1/latest/model").rglob("*.json")}
    (DEST/"submissions/latest/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"destination": str(DEST), "tasks": manifest["tasks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
