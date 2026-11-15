"""核对实验产物，并可从零重训全量森林检查确定性。

示例：python -m task1.verify --run task1/outputs/baseline --retrain-forest
所有检查结论写入该运行目录 verification.json，便于阶段记录引用。
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import WEATHER, Preprocessor, file_hash, read_data, stratified_folds
from .metrics import classification_metrics
from .models import RandomForestClassifier, model_from_dict
from .run import now, write_json


def require(condition, message):
    """不用 assert：即使 Python 开启 -O 优化，也必须执行复现检查。"""
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description="核对 Task 1 的指标、数据与复现结果")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--retrain-forest", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "completed", "不能核验未完成的运行。")
    data_folder = Path(config["data_dir"])
    train, test, sample = read_data(data_folder)
    for name, expected in manifest["data_sha256"].items():
        require(file_hash(data_folder / f"{name}.csv") == expected, f"原始数据 {name} 已变更。")
    expected_folds = stratified_folds(train.label.to_numpy(), config["folds"], config["seed"])
    stored_folds = pd.read_csv(args.run / "folds.csv")
    require(np.array_equal(stored_folds.id, train.id), "折号文件 id 错位。")
    require(np.array_equal(stored_folds.fold, expected_folds + 1), "折号无法由种子重现。")
    evidence = {"verified_at": now(), "data_hashes_match": True, "folds_reproduced": True,
                "checks": [], "models": {}}
    summaries = json.loads((args.run / "comparison.json").read_text(encoding="utf-8"))
    for summary in summaries:
        name = summary["name"]
        folder = args.run / name
        oof = pd.read_csv(folder / "oof_predictions.csv")
        require(np.array_equal(oof.id, train.id), f"{name} 的 OOF id 错位。")
        require(np.array_equal(oof.actual, train.label), f"{name} 的 OOF 真实标签错位。")
        require(np.array_equal(oof.fold, expected_folds + 1), f"{name} 的 OOF 折号错位。")
        probability = oof[[f"probability_{i}" for i in range(1, 5)]].to_numpy()
        require(np.isfinite(probability).all() and (probability >= 0).all(), "OOF 概率不合法。")
        require(np.allclose(probability.sum(axis=1), 1.0), "OOF 概率和不为 1。")
        require(np.array_equal(oof.predicted, probability.argmax(axis=1) + 1), "OOF 概率与预测不一致。")
        recalculated = classification_metrics(oof.actual, oof.predicted)
        require(abs(recalculated["macro_f1"] - summary["oof_metrics"]["macro_f1"]) < 1e-12,
                "OOF Macro-F1 不能从记录重算。")
        require(recalculated["confusion_matrix"] == summary["oof_metrics"]["confusion_matrix"],
                "混淆矩阵不能从记录重算。")
        fold_scores = []
        for fold in range(config["folds"]):
            part = oof[oof.fold == fold + 1]
            score = classification_metrics(part.actual, part.predicted)["macro_f1"]
            require(abs(score - summary["folds"][fold]["macro_f1"]) < 1e-12, "逐折分数无法重算。")
            fold_scores.append(score)
        require(abs(np.mean(fold_scores) - summary["cv_mean_macro_f1"]) < 1e-12, "CV 均值不一致。")

        state = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        prep = Preprocessor.from_dict(state["preprocessor"])
        x = prep.transform(test)
        prediction = model_from_dict(state["model"]).predict(x)
        integer = pd.read_csv(folder / "submission_integer.csv")
        weather = pd.read_csv(folder / "submission_weather.csv")
        for frame in [integer, weather]:
            require(list(frame.columns) == ["id", "label"], "提交字段错误。")
            require(np.array_equal(frame.id, test.id), "提交 id 顺序错误。")
            require(len(frame) == len(test), "提交记录数错误。")
        require(np.array_equal(integer.label, prediction), "独立加载模型的预测与提交不一致。")
        require(weather.label.tolist() == [WEATHER[int(i)] for i in prediction], "两种标签编码不等价。")
        record = {"oof_metrics_recalculated": True, "submission_matches_loaded_model": True,
                  "integer_sha256": file_hash(folder / "submission_integer.csv"),
                  "weather_sha256": file_hash(folder / "submission_weather.csv")}
        if name == "forest" and args.retrain_forest:
            # 从原始训练数据和公开配置重新训练，不能复用旧模型里的树或预处理参数。
            started = time.perf_counter()
            fresh_prep = Preprocessor(standardize=False, anomaly_flags=config["anomaly_flags"]).fit(train)
            fresh_model = RandomForestClassifier(config["trees"], config["forest_depth"],
                                                  config["forest_leaf"], config["seed"])
            fresh_model.fit(fresh_prep.transform(train), train.label.to_numpy())
            require(fresh_prep.to_dict() == state["preprocessor"], "重训预处理参数不一致。")
            require(fresh_model.to_dict() == state["model"], "相同配置重训的森林节点不一致。")
            require(np.array_equal(fresh_model.predict(fresh_prep.transform(test)), prediction),
                    "相同配置重训的预测不一致。")
            record.update(full_forest_retraining_identical=True,
                          retraining_verification_seconds=time.perf_counter() - started)
        evidence["models"][name] = record
        print(f"{name}: 数据、折号、OOF 指标、提交文件与加载预测检查通过。", flush=True)
    evidence["status"] = "passed"
    write_json(args.run / "verification.json", evidence)
    print(f"核验记录：{args.run / 'verification.json'}")


if __name__ == "__main__":
    main()
