"""Task 1 实验入口：检查数据、固定交叉验证、全量训练并导出预测。

从项目根目录执行：python -m task1.run
每次运行使用独立目录，保留原始数据哈希、配置、逐折指标和模型，
以便报告中的每项结论都能够追溯到实际实验。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import (FEATURES, NUMERIC, CATEGORICAL, WEATHER, Preprocessor,
                   file_hash, inner_holdout, read_data, stratified_folds)
from .metrics import classification_metrics
from .models import (CARTClassifier, RandomForestClassifier, SoftmaxRegression,
                     model_from_dict)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """JSON 保留中文，拒绝 NaN，以保证配置和模型在其他环境也能读取。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def audit_data(train, test, sample, folder):
    """将数据检查固化为机器可读结果；异常值仅统计，不删除。"""
    result = {"files": {}, "label_counts": {str(k): int(v) for k, v in
              train.label.value_counts().sort_index().items()},
              "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
              "submission_sample_labels": sample.label.unique().tolist(),
              "submission_format_note": "训练标签为整数，样例为天气名称；同时导出两种版本，上传前核对 Kaggle。"}
    for name, frame in [("train", train), ("test", test), ("sample_submission", sample)]:
        item = {"rows": len(frame), "columns": frame.columns.tolist(),
                "sha256": file_hash(folder / f"{name}.csv"),
                "missing": {col: int(count) for col, count in frame.isna().sum().items()},
                "unique_ids": bool(frame.id.is_unique)}
        if name != "sample_submission":
            item["duplicate_features"] = int(frame.duplicated(FEATURES).sum())
            item["anomalies"] = {
                "humidity_gt100": int((frame.Humidity > 100).sum()),
                "precipitation_gt100": int((frame["Precipitation (%)"] > 100).sum()),
                "temperature_gt60": int((frame.Temperature > 60).sum()),
                "pressure_outside900_1100": int(((frame["Atmospheric Pressure"] < 900) |
                                                (frame["Atmospheric Pressure"] > 1100)).sum())}
        result["files"][name] = item
    result["cross_dataset_duplicate_features"] = int(len(train[FEATURES].merge(test[FEATURES], on=FEATURES)))
    result["unseen_test_categories"] = {
        col: sorted(set(test[col].dropna()) - set(train[col].dropna())) for col in CATEGORICAL}
    return result


def create_model(name, args, seed, epochs=None):
    if name == "softmax":
        return SoftmaxRegression(learning_rate=args.learning_rate, l2=args.l2,
                                 epochs=args.epochs if epochs is None else epochs,
                                 patience=args.patience)
    if name == "cart":
        return CARTClassifier(max_depth=args.cart_depth, min_leaf=args.cart_leaf, seed=seed)
    return RandomForestClassifier(n_trees=args.trees, max_depth=args.forest_depth,
                                  min_leaf=args.forest_leaf, seed=seed)


def select_epochs(frame, labels, args, seed):
    """只在外层训练集内部选轮数，内层预处理也只拟合内层训练集。"""
    fit_indices, valid_indices = inner_holdout(labels, seed=seed)
    prep = Preprocessor(standardize=True, anomaly_flags=args.anomaly_flags).fit(frame.iloc[fit_indices])
    fit_x = prep.transform(frame.iloc[fit_indices])
    valid_x = prep.transform(frame.iloc[valid_indices])
    model = create_model("softmax", args, seed)
    model.fit(fit_x, labels[fit_indices], validation=(valid_x, labels[valid_indices]))
    return model.best_epoch, model.history


def export_submission(test, predictions, folder):
    """输出两种标签编码；两者预测完全相同，只是序列化的标签形式不同。"""
    predictions = np.asarray(predictions, dtype=int)
    if len(predictions) != len(test) or not np.isin(predictions, [1, 2, 3, 4]).all():
        raise ValueError("预测数量或标签范围不正确。")
    pd.DataFrame({"id": test.id.to_numpy(), "label": predictions}).to_csv(
        folder / "submission_integer.csv", index=False)
    pd.DataFrame({"id": test.id.to_numpy(), "label": [WEATHER[int(v)] for v in predictions]}).to_csv(
        folder / "submission_weather.csv", index=False)


def evaluate_model(name, train, test, fold_id, args, out):
    """每个模型使用完全相同的外层划分，预处理在每折中重新拟合。"""
    folder = out / name
    folder.mkdir()
    y = train.label.to_numpy(dtype=int)
    oof_probability = np.zeros((len(train), 4))
    folds, chosen_epochs = [], []
    for fold in range(args.folds):
        started = time.perf_counter()
        fit_indices, valid_indices = np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold)
        fit_frame, valid_frame = train.iloc[fit_indices], train.iloc[valid_indices]
        epochs = None
        if name == "softmax":
            epochs, history = select_epochs(fit_frame, y[fit_indices], args, args.seed + fold)
            chosen_epochs.append(epochs)
            write_json(folder / f"fold_{fold + 1}_inner_history.json", history)
        prep = Preprocessor(standardize=name == "softmax", anomaly_flags=args.anomaly_flags).fit(fit_frame)
        fit_x, valid_x = prep.transform(fit_frame), prep.transform(valid_frame)
        prep_seconds = time.perf_counter() - started
        model = create_model(name, args, args.seed + fold, epochs)
        fit_started = time.perf_counter()
        model.fit(fit_x, y[fit_indices])
        fit_seconds = time.perf_counter() - fit_started
        inference_started = time.perf_counter()
        probabilities = model.predict_proba(valid_x)
        inference_seconds = time.perf_counter() - inference_started
        oof_probability[valid_indices] = probabilities
        metrics = classification_metrics(y[valid_indices], probabilities.argmax(axis=1) + 1)
        metrics.update(fold=fold + 1, train_rows=len(fit_indices), valid_rows=len(valid_indices),
                       preparation_and_inner_selection_seconds=prep_seconds,
                       fit_seconds=fit_seconds, predict_seconds=inference_seconds,
                       selected_epochs=epochs)
        folds.append(metrics)
        write_json(folder / f"fold_{fold + 1}_metrics.json", metrics)
        print(f"{name} fold {fold + 1}/{args.folds}: Macro-F1={metrics['macro_f1']:.6f}, "
              f"train={fit_seconds:.2f}s", flush=True)
    oof_predictions = oof_probability.argmax(axis=1) + 1
    oof = pd.DataFrame({"id": train.id, "fold": fold_id + 1, "actual": y, "predicted": oof_predictions})
    for label in range(1, 5):
        oof[f"probability_{label}"] = oof_probability[:, label - 1]
    oof.to_csv(folder / "oof_predictions.csv", index=False)
    scores = np.array([fold["macro_f1"] for fold in folds])
    summary = {"name": name, "cv_mean_macro_f1": float(scores.mean()),
               "cv_std_macro_f1": float(scores.std(ddof=1)),
               "oof_metrics": classification_metrics(y, oof_predictions), "folds": folds}

    # 全量模型的训练轮数来自内层选择结果的中位数，不查看外层标签表现决定轮数。
    full_epochs = int(np.median(chosen_epochs)) if chosen_epochs else None
    full_started = time.perf_counter()
    prep = Preprocessor(standardize=name == "softmax", anomaly_flags=args.anomaly_flags).fit(train)
    full_x = prep.transform(train)
    prep_seconds = time.perf_counter() - full_started
    model = create_model(name, args, args.seed, full_epochs)
    fit_started = time.perf_counter()
    model.fit(full_x, y)
    fit_seconds = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    test_x = prep.transform(test)
    probabilities = model.predict_proba(test_x)
    predictions = probabilities.argmax(axis=1) + 1
    inference_seconds = time.perf_counter() - inference_started
    artifact = {"schema_version": 1, "classes": WEATHER, "seed": args.seed,
                "train_sha256": file_hash(Path(args.data_dir) / "train.csv"),
                "preprocessor": prep.to_dict(), "model": model.to_dict()}
    write_json(folder / "model.json", artifact)
    if hasattr(model, "history"):
        write_json(folder / "full_training_history.json", model.history)
    export_submission(test, predictions, folder)

    # 实际从磁盘重新读取模型，并要求概率与标签一致，避免仅能复现内存里的结果。
    loaded = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    loaded_x = Preprocessor.from_dict(loaded["preprocessor"]).transform(test)
    loaded_prob = model_from_dict(loaded["model"]).predict_proba(loaded_x)
    if not np.allclose(probabilities, loaded_prob, rtol=0, atol=1e-12):
        raise RuntimeError("模型保存后重新加载，概率发生变化。")
    if not np.array_equal(predictions, loaded_prob.argmax(axis=1) + 1):
        raise RuntimeError("模型保存后重新加载，预测标签发生变化。")
    importance = None
    if name == "cart":
        importance = model.importance_raw
    elif name == "forest":
        importance = np.sum([tree.importance_raw for tree in model.trees], axis=0)
    if importance is not None:
        total = importance.sum()
        importance = importance / total if total > 0 else importance
        pd.DataFrame({"feature": prep.feature_names, "gini_importance": importance}).sort_values(
            "gini_importance", ascending=False).to_csv(folder / "feature_importance.csv", index=False)
    summary["full_training"] = {"preprocessing_seconds": prep_seconds, "fit_seconds": fit_seconds,
                                 "total_offline_seconds": prep_seconds + fit_seconds,
                                 "all_test_inference_seconds_including_transform": inference_seconds,
                                 "epochs": full_epochs, "reload_verified": True,
                                 "prediction_counts": {str(k): int(v) for k, v in
                                     pd.Series(predictions).value_counts().sort_index().items()}}
    write_json(folder / "summary.json", summary)
    return summary


def write_report(out, summaries, audit, args):
    """生成当前运行的中文实验报告；时间是本次机器实测值，不作跨机器承诺。"""
    lines = ["# Task 1 实验结果", "", f"记录时间（Asia/Shanghai）：{now()}", "",
             "## 实验协议", "", f"- 随机种子：{args.seed}；外层分层 {args.folds} 折。",
             "- 固定同一组折号；每折预处理仅在该折训练部分拟合。",
             "- Softmax 在训练折内部留出 15% 选择轮数，再从零拟合整个外层训练折。",
             "- 不使用 id，不删除异常记录；异常标记是否启用见 config.json。",
             "- 此结果用于模型开发和选择；选出最优模型后的同一 CV 分数并非独立测试成绩。",
             "- 未上传 Kaggle，因此没有公开榜或私有榜成绩。", "",
             "## 模型比较", "", "| 模型 | CV Macro-F1 均值 | 样本标准差 | OOF Macro-F1 | 全量训练秒 | 全测试推理秒 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        timing = row["full_training"]
        lines.append(f"| {row['name']} | {row['cv_mean_macro_f1']:.6f} | {row['cv_std_macro_f1']:.6f} "
                     f"| {row['oof_metrics']['macro_f1']:.6f} | {timing['fit_seconds']:.3f} "
                     f"| {timing['all_test_inference_seconds_including_transform']:.3f} |")
    best = max(summaries, key=lambda row: row["cv_mean_macro_f1"])
    lines += ["", f"当前运行中 CV 均值最高的候选：**{best['name']}**。", "",
              "## 最优候选各类 OOF 指标", "", "| 类别 | Precision | Recall | F1 | 样本数 |",
              "|---|---:|---:|---:|---:|"]
    metrics = best["oof_metrics"]
    for i, name in WEATHER.items():
        lines.append(f"| {i}: {name} | {metrics['precision'][i-1]:.4f} | {metrics['recall'][i-1]:.4f} "
                     f"| {metrics['f1'][i-1]:.4f} | {metrics['support'][i-1]} |")
    lines += ["", "混淆矩阵：行是真实类别，列是预测类别；顺序均为 Sunny、Cloudy、Rainy、Snowy。", "", "```text"]
    lines += [str(row) for row in metrics["confusion_matrix"]]
    lines += ["```", "", "## 输入检查与限制", "",
              f"- 训练样本：{audit['files']['train']['rows']}；测试样本：{audit['files']['test']['rows']}。",
              f"- 训练标签计数：{audit['label_counts']}。",
              "- 整数标签与样例天气名称的格式差异尚需通过 Kaggle 核对；两种编码均已导出。",
              "- 各模型的预测文件位于各自目录，未自动选定 Kaggle 最终提交。",
              "- 森林为单机顺序训练；耗时包含本次系统负载影响。",
              "- Gini 重要性仅描述树分裂贡献，不代表因果关系。", "",
              "## 可复现证据", "", "- manifest.json：环境信息、原始数据哈希、运行状态。",
              "- config.json：所有命令行配置。", "- folds.csv：每条训练记录的外层折号。",
              "- 各模型目录：逐折指标、OOF 概率、全量模型、训练历史和预测文件。",
              "- 每个模型均完成保存、重新加载、概率和预测一致性检查。", ""]
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="手写天气分类模型实验")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "comp-5434-20261-project-task-1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", nargs="+", choices=["softmax", "cart", "forest"],
                        default=["softmax", "cart", "forest"])
    parser.add_argument("--seed", type=int, default=5434)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--cart-depth", type=int, default=12)
    parser.add_argument("--cart-leaf", type=int, default=5)
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--forest-depth", type=int, default=16)
    parser.add_argument("--forest-leaf", type=int, default=3)
    parser.add_argument("--anomaly-flags", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.trees < 1 or args.cart_leaf < 1 or args.forest_leaf < 1:
        parser.error("训练轮数、耐心值、树数量和叶子样本数必须为正数。")
    if args.learning_rate <= 0 or args.l2 < 0 or args.cart_depth < 0 or args.forest_depth < 0:
        parser.error("学习率必须为正，正则化与深度不能为负。")
    if len(set(args.models)) != len(args.models):
        parser.error("模型名称不能重复。")
    out = args.output_dir or ROOT / "task1" / "outputs" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(out / "config.json", config)
    manifest = {"started_at": now(), "status": "running", "python": sys.version,
                "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(),
                "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
                "processor_note": "platform.processor 为系统架构信息，不保证是商品 CPU 型号。",
                "tzdata": importlib.metadata.version("tzdata"),
                "thread_environment": {key: os.environ.get(key) for key in
                                       ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
                "numpy_build_dependencies": np.__config__.CONFIG.get("Build Dependencies", {}),
                "source_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    write_json(out / "manifest.json", manifest)
    try:
        train, test, sample = read_data(args.data_dir)
        audit = audit_data(train, test, sample, args.data_dir)
        write_json(out / "data_audit.json", audit)
        manifest["data_sha256"] = {k: v["sha256"] for k, v in audit["files"].items()}
        fold_id = stratified_folds(train.label.to_numpy(), args.folds, args.seed)
        pd.DataFrame({"id": train.id, "fold": fold_id + 1}).to_csv(out / "folds.csv", index=False)
        summaries = []
        for name in args.models:
            summaries.append(evaluate_model(name, train, test, fold_id, args, out))
            write_json(out / "comparison.json", summaries)
        write_report(out, summaries, audit, args)
        manifest.update(status="completed", finished_at=now())
        write_json(out / "manifest.json", manifest)
        print(f"完成。实验目录：{out}", flush=True)
    except Exception as error:
        manifest.update(status="failed", finished_at=now(), error=repr(error))
        write_json(out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
