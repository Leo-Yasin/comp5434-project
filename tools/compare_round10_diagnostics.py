"""比较两台电脑收集的Round10诊断报告，不训练、不重新预测。

此工具寻找最早出现数值差异的已采样阶段。探针相同不代表全部10000行相同；
差异也不能直接换算成Kaggle RMSE，因为诊断报告不包含测试集真实评分。
"""
import argparse
import json
import math
from pathlib import Path


def flatten(value):
    """兼容一维评分、二维logits和CLS，仅展开数值叶节点。"""
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield float(value)


def compare_values(left, right):
    """按ID对齐探针；保留缺失信息，不把缺失数据当作零差异。"""
    if not left or not right or "error" in left or "error" in right:
        return {"status": "缺少可比较的数值探针"}
    a = {v: i for i, v in enumerate(left["probe_ids"])}
    b = {v: i for i, v in enumerate(right["probe_ids"])}
    common = [v for v in a if v in b]
    result = {"common_probe_ids": common}
    for key in ("logits", "prediction", "cls"):
        if key not in left or key not in right:
            continue
        errors = []
        for id_ in common:
            x = list(flatten(left[key][a[id_]]))
            y = list(flatten(right[key][b[id_]]))
            if len(x) != len(y):
                raise ValueError(f"ID={id_}的{key}形状不一致。")
            if not all(math.isfinite(v) for v in x + y):
                raise ValueError("探针包含非有限数值。")
            errors.extend(abs(u - v) for u, v in zip(x, y))
        if errors:
            result[key] = {
                "scalars": len(errors),
                "mean_absolute_difference": sum(errors) / len(errors),
                "maximum_absolute_difference": max(errors),
                "root_mean_square_difference": math.sqrt(sum(v*v for v in errors)/len(errors)),
            }
    return result


def compare(left, right):
    report = {"test_sha256_equal": left["test_sha256"] == right["test_sha256"]}
    report["model_manifest_sha256_equal"] = (
        left["model_integrity"]["manifest_sha256"] == right["model_integrity"]["manifest_sha256"]
    )
    report["model_mismatches"] = {
        "left": left["model_integrity"]["mismatches"],
        "right": right["model_integrity"]["mismatches"],
    }
    sources = set(left["source_sha256_lf"]) | set(right["source_sha256_lf"])
    report["different_source_files"] = sorted(
        key for key in sources if left["source_sha256_lf"].get(key) != right["source_sha256_lf"].get(key)
    )
    report["environments"] = {}
    for role in ("numeric", "semantic", "english"):
        a, b = left["environments"][role], right["environments"][role]
        # 路径不同属于正常现象；重点展示版本、体系结构和构建信息。
        report["environments"][role] = {
            key: {"left": a.get(key), "right": b.get(key)}
            for key in ("python", "machine", "processor", "packages", "onnxruntime_build_info")
            if a.get(key) != b.get(key)
        }
    report["stages"] = {}
    for stage in ("embedding", "english", "adapted", "title"):
        a, b = left["stages"].get(stage, {}), right["stages"].get(stage, {})
        report["stages"][stage] = {
            "state_equal": a.get(".state.json") == b.get(".state.json"),
            "values": compare_values(a.get("values"), b.get("values")),
        }
    report["parent_values"] = compare_values(left.get("parent_values"), right.get("parent_values"))
    report["prediction_comparisons"] = {
        "left": left.get("prediction_comparison"), "right": right.get("prediction_comparison")
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (args.left, args.right)]
    result = compare(*reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已保存对照报告：", args.output)


if __name__ == "__main__":
    main()
