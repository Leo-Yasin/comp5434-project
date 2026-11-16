"""带运行环境断点保护的Round10预测入口，不更改原模型的数值计算。

第一次必须使用空工作目录，随后断点续跑必须保持输入、权重、源码及三环境一致。
此保护避免跨环境误用旧向量；不承诺不同CPU体系结构的冷启动数值必然一致。
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "test", "semantic-python", "english-python", "work-dir", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="只检查依赖和绑定环境，不运行任何编码或预测。")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖现有预测：{args.output}")
    if args.model.resolve() == args.work_dir.resolve():
        raise ValueError("工作目录不能与模型目录相同。")
    marker = args.work_dir / "checked_environment.json"
    existing = {path.name for path in args.work_dir.iterdir()} if args.work_dir.exists() else set()
    if existing - {"environment_candidate.json"} and not marker.exists():
        raise ValueError("该目录已有未经本入口绑定的旧缓存，请另选空工作目录；不必删除原诊断数据。")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    # 收集器只读取现有文件，不执行编码。先写独立临时报告，核验后才更新环境绑定。
    candidate = args.work_dir / "environment_candidate.json"
    command = [sys.executable, "-X", "utf8", str(ROOT / "tools/collect_round10_diagnostics.py")]
    for key in ("model", "test", "semantic_python", "english_python", "work_dir"):
        command += ["--" + key.replace("_", "-"), str(getattr(args, key).resolve())]
    command += ["--prediction", str(args.output.resolve()), "--output", str(candidate.resolve())]
    subprocess.run(command, cwd=ROOT, check=True)
    report = json.loads(candidate.read_text(encoding="utf-8"))
    if report["model_integrity"]["mismatches"]:
        raise ValueError("模型逐文件SHA不匹配，请重新校验模型包。")
    if any("error" in env for env in report["environments"].values()):
        raise ValueError("至少一个Python环境查询失败，请查看environment_candidate.json。")
    required = {
        "numeric": ("numpy", "pandas", "scipy", "scikit-learn"),
        "semantic": ("numpy", "pandas", "tokenizers"),
        "english": ("numpy", "pandas", "openvino", "torch", "transformers", "tokenizers", "sentencepiece"),
    }
    missing = [f"{role}:{name}" for role, names in required.items() for name in names
               if report["environments"][role]["packages"].get(name) is None]
    if "onnxruntime_version" not in report["environments"]["semantic"]:
        missing.append("semantic:onnxruntime")
    if missing:
        raise ValueError(f"推理依赖缺失{missing}，尚未启动编码，请按对应环境依赖表安装。")
    # 包元数据存在不代表实际能导入。导入真实入口，提前发现Dask/six等依赖或ABI错误；
    # 这些模块的main均有入口保护，单纯import不会训练、预测或加载模型权重。
    for python, modules in (
        (sys.executable, "task2.round10_predict"),
        (args.semantic_python, "task2.embedding_infer"),
        (args.english_python, "task2.english_infer, task2.partial9_infer"),
    ):
        subprocess.run([str(python), "-X", "utf8", "-c", "import " + modules], cwd=ROOT, check=True)
    signature = {key: report[key] for key in ("environments", "test_sha256", "source_sha256_lf", "model_integrity")}
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        changed = [key for key in signature if previous.get(key) != signature[key]]
        if changed:
            raise ValueError(f"断点指纹变化{changed}，拒绝混用缓存，请另选空工作目录。")
    else:
        marker.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.check_only:
        print("依赖、模型SHA及环境断点绑定检查通过；尚未执行预测。", flush=True)
        return
    # 采用同一数值解释器和同一原始入口；包装器不替换logits、向量或最终预测。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-X", "utf8", "-m", "task2.round10_predict"]
    for key in ("model", "test", "semantic_python", "english_python", "work_dir", "output"):
        command += ["--" + key.replace("_", "-"), str(getattr(args, key).resolve())]
    subprocess.run(command, cwd=ROOT, check=True)
    print("预测完成；请对照原CSV核验数值，环境绑定本身不证明Kaggle分数。", flush=True)


if __name__ == "__main__":
    main()
