"""汇总首轮对比、异常标记消融与 OOF 错误，生成可用于报告的中文记录。

默认对应本次 baseline / forest_flags 两个运行；也可以传入其他运行目录。
此脚本只分析训练集的折外预测，不读取测试集真实标签。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run import now, write_json


def main():
    parser = argparse.ArgumentParser(description="汇总 Task 1 实验及错误分析")
    parser.add_argument("--baseline", type=Path, default=Path("task1/outputs/baseline"))
    parser.add_argument("--flags", type=Path, default=Path("task1/outputs/forest_flags"))
    parser.add_argument("--output", type=Path, default=Path("task1/docs/实验总结.md"))
    args = parser.parse_args()
    baseline = json.loads((args.baseline / "comparison.json").read_text(encoding="utf-8"))
    flags = json.loads((args.flags / "comparison.json").read_text(encoding="utf-8"))
    config = json.loads((args.baseline / "config.json").read_text(encoding="utf-8"))
    flags_config = json.loads((args.flags / "config.json").read_text(encoding="utf-8"))
    # 消融比较要求相同数据和划分。其余森林配置也逐项核对，防止混入其他改动。
    for key in ["data_dir", "seed", "folds", "trees", "forest_depth", "forest_leaf"]:
        if config[key] != flags_config[key]:
            raise ValueError(f"消融比较的 {key} 不一致。")
    if config["anomaly_flags"] or not flags_config["anomaly_flags"]:
        raise ValueError("消融应当比较不加标记和加入标记两种方案。")
    first_folds = pd.read_csv(args.baseline / "folds.csv")
    second_folds = pd.read_csv(args.flags / "folds.csv")
    if not first_folds.equals(second_folds):
        raise ValueError("消融实验的逐样本折号不同。")
    forest = next(row for row in baseline if row["name"] == "forest")
    flags_forest = next(row for row in flags if row["name"] == "forest")
    train = pd.read_csv(Path(config["data_dir"]) / "train.csv")
    oof = pd.read_csv(args.baseline / "forest" / "oof_predictions.csv")
    if not np.array_equal(train.id, oof.id):
        raise ValueError("OOF 与训练记录没有对齐。")
    anomaly = ((train.Humidity > 100) | (train["Precipitation (%)"] > 100) |
               (train.Temperature > 60) | (train["Atmospheric Pressure"] < 900) |
               (train["Atmospheric Pressure"] > 1100))
    correct = oof.actual == oof.predicted
    analysis = train.copy()
    analysis["oof_prediction"] = oof.predicted
    analysis["oof_fold"] = oof.fold
    analysis["oof_confidence"] = oof[[f"probability_{i}" for i in range(1, 5)]].max(axis=1)
    analysis["has_anomaly_flag"] = anomaly
    errors = analysis.loc[~correct].sort_values("oof_confidence", ascending=False)
    errors.to_csv(args.baseline / "forest" / "oof_errors.csv", index=False)
    difference = flags_forest["cv_mean_macro_f1"] - forest["cv_mean_macro_f1"]
    lines = ["# Task 1 首轮实验与消融总结", "", f"更新时间：{now()}", "",
             "## 已完成工作", "", "- 数据审计、相同的分层 5 折验证及训练折内预处理。",
             "- Softmax、CART、随机森林的自主实现，均有详细中文注释。",
             "- 每折分数、OOF 概率、混淆矩阵、训练历史、全量模型和两种编码提交文件。",
             "- 异常标记的单因素方案对比，未使用 Kaggle 测试标签。", "",
             "## 效果比较", "", "| 方案 | CV Macro-F1 均值 | 折间样本标准差 |",
             "|---|---:|---:|"]
    for row in baseline:
        lines.append(f"| {row['name']} | {row['cv_mean_macro_f1']:.6f} | {row['cv_std_macro_f1']:.6f} |")
    lines += [f"| forest + 异常标记 | {flags_forest['cv_mean_macro_f1']:.6f} | {flags_forest['cv_std_macro_f1']:.6f} |",
              "", f"异常标记的均值变化：{difference:+.6f}。本次没有观察到平均分提升，暂保留不加标记的森林。",
              "差异较小，未进行统计显著性检验，不能据此断言异常标记在所有划分下都无效。",
              "增加列也会改变随机特征子集的具体抽样，该比较反映整个方案的效果。", "",
              "## 结果解释", "",
              "- 树模型优于线性基线，支持进一步研究特征阈值与非线性交互的价值。",
              "- 森林比单树的分数更高、折间波动更小；这只是本次配置与划分的观察。",
              "- Softmax 的五次内层选择都达到 1000 轮上限，不能声称已经充分收敛或是线性模型的最佳成绩。",
              "- 基础森林 OOF 中 Cloudy 的 F1 最低，与 Rainy 之间的混淆值得继续检查。", "",
              "## 错误分析", "", f"- 基础森林共错分 {int((~correct).sum())} / {len(train)} 条 OOF 样本。",
              f"- 满足至少一个异常标记的 {int(anomaly.sum())} 条样本，OOF accuracy 为 {correct[anomaly].mean():.6f}。",
              f"- 其余 {int((~anomaly).sum())} 条样本，OOF accuracy 为 {correct[~anomaly].mean():.6f}。",
              "- 以上是按事先定义阈值的描述性分组；不能仅凭低准确率认定标签错误，也不能认定是不可消除噪声。",
              "- 详细错分记录已保存为基础森林目录下 oof_errors.csv，按 OOF 置信度从高到低排列。", "",
              "## 当前候选与限制", "",
              "当前候选为 baseline/forest/model.json，提交文件同目录。整数与天气名称版本需先核对 Kaggle 要求。",
              "这些是用于开发和选择的本地 CV 分数，不是 Kaggle 成绩或独立保留测试成绩。",
              "准确率提升应结合重复划分或独立评估进一步确认，不继续无上限追逐同一组 CV。", "",
              "## 后续阶段", "",
              "1. 核对 Kaggle 标签编码并提交当前候选，记录公开榜分数。",
              "2. 按需比较叶子限制与树深度，所有尝试记录配置和结果。",
              "3. 将确认后的结果整理进课程最终报告；与 Task 2 合并制作展示材料。", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    write_json(args.baseline / "forest" / "error_analysis.json", {
        "oof_errors": int((~correct).sum()), "anomaly_count": int(anomaly.sum()),
        "anomaly_accuracy": float(correct[anomaly].mean()),
        "other_accuracy": float(correct[~anomaly].mean()),
        "flags_cv_change": difference,
    })
    print(f"已生成：{args.output}")


if __name__ == "__main__":
    main()
