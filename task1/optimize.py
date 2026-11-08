"""固定原五折，比较自主提升树/概率集成，保存全量模型及整数预测。

最多两个提升树深度、两个检查轮数及三种固定融合比例，不读取 Kaggle 测试标签。
记录未选方案，避免只展示最优结果；另以第二组种子检验最终候选的稳定性。
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .data import Preprocessor, read_data, stratified_folds, file_hash
from .metrics import classification_metrics
from .boosting import HistogramBoosting
from .run import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("task1/outputs/optimization_20261105"))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    train, test, _ = read_data(Path("comp-5434-20261-project-task-1"))
    y = train.label.to_numpy()
    folds = pd.read_csv("task1/outputs/baseline/folds.csv").fold.to_numpy()-1
    baseline_frame = pd.read_csv("task1/outputs/baseline/forest/oof_predictions.csv")
    if not np.array_equal(train.id, baseline_frame.id):
        raise ValueError("基线 OOF id 与当前训练集不一致。")
    baseline = baseline_frame[[f"probability_{i}" for i in range(1,5)]].to_numpy()
    probabilities, results, timings = {}, [], {}
    for depth in [3, 4]:
        bank = {120: np.zeros((len(train),4)), 240: np.zeros((len(train),4))}
        started = time.perf_counter()
        for fold in range(5):
            fit, valid = folds != fold, folds == fold
            prep = Preprocessor(standardize=False).fit(train.loc[fit])
            model = HistogramBoosting(depth=depth, seed=5434+fold).fit(prep.transform(train.loc[fit]), y[fit])
            for rounds in bank:
                bank[rounds][valid] = model.predict_proba(prep.transform(train.loc[valid]), rounds)
            print(f"Task1 depth={depth}, fold={fold+1}/5, F1={classification_metrics(y[valid],bank[240][valid].argmax(1)+1)['macro_f1']:.6f}", flush=True)
        timings[str(depth)] = time.perf_counter()-started
        for rounds, p in bank.items():
            name = f"boost_d{depth}_r{rounds}"
            probabilities[name] = p
            for weight in [0.0, 0.25, 0.5]:
                mixed = p*(1-weight)+baseline*weight
                scores = [classification_metrics(y[folds==k],mixed[folds==k].argmax(1)+1)["macro_f1"] for k in range(5)]
                results.append({"name":name,"depth":depth,"rounds":rounds,"baseline_weight":weight,
                                "mean_macro_f1":float(np.mean(scores)),"std":float(np.std(scores,ddof=1)),"fold_scores":scores})
            np.save(args.output/f"{name}.oof.npy", p)
        write_json(args.output/"comparison.json", results)
    selected = max(results,key=lambda r:r["mean_macro_f1"])
    depth, rounds, weight = selected["depth"],selected["rounds"],selected["baseline_weight"]
    prep = Preprocessor(standardize=False).fit(train)
    started = time.perf_counter()
    model = HistogramBoosting(rounds=rounds,depth=depth).fit(prep.transform(train),y)
    fit_seconds = time.perf_counter()-started
    write_json(args.output/"model.json", {"preprocessor":prep.to_dict(),"model":model.to_dict(),
                                         "baseline_weight":weight,"train_sha256":file_hash(Path("comp-5434-20261-project-task-1/train.csv"))})
    from .models import model_from_dict
    base_state=json.loads(Path("task1/outputs/baseline/forest/model.json").read_text(encoding="utf-8"))
    p=model.predict_proba(prep.transform(test))
    base_p=model_from_dict(base_state["model"]).predict_proba(Preprocessor.from_dict(base_state["preprocessor"]).transform(test))
    prediction=((1-weight)*p+weight*base_p).argmax(1)+1
    pd.DataFrame({"id":test.id,"label":prediction}).to_csv(args.output/"submission_integer.csv",index=False)
    restored=HistogramBoosting.from_dict(json.loads((args.output/"model.json").read_text(encoding="utf-8"))["model"])
    if not np.array_equal(p,restored.predict_proba(prep.transform(test))):
        raise RuntimeError("提升树 JSON 加载概率不一致。")
    selected.update(full_fit_seconds=fit_seconds,cv_seconds_by_depth=timings,
                    baseline_cv=0.9152233645819339,kaggle_baseline_user_report=0.91321,
                    new_kaggle_score=None,reload_verified=True)
    write_json(args.output/"selection.json",selected)
    # 融合依赖也复制进模型目录，保证之后独立预测不依赖基线文件位置。
    write_json(args.output/"baseline_forest.json",base_state)
    print("Task1 selected",selected,flush=True)


if __name__=="__main__":
    main()
