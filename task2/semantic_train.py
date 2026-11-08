"""第五轮大改：语义分布 + 显式噪声通道似然 + 自主非线性融合。

内部选择与冻结确认分开；只在内部组比较两种架构及预定融合权重，不根据
Kaggle 分数或确认组分数再改参数。所有外部语义输入都来自冻结模型推理。
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import ROOT, write_json
from .round3_train import evaluate
from .semantic_fusion import semantic_features, fit_linear, prior_predictions, known_history_mask
from .calibration import adjusted_risk
from .noise_softmax import fit_channel, predict_channel
from .regression_boost import ResidualHistogramBoost

OUT = ROOT/"task2/outputs/round5_20261105"


def load_development():
    original = dict(np.load(ROOT/"task2/outputs/round3_20261105/development.npz"))
    selection = dict(np.load(OUT/"semantic_selection.npz"))
    idx = selection["index"]
    n = len(original["y"])
    data = {key: value[idx] if value.ndim and value.shape[0] == n else value for key, value in original.items()}
    semantics = dict(np.load(OUT/"semantic_development.npz"))
    if not np.array_equal(semantics["id"], data["id"]):
        raise ValueError("语义开发预测错位。")
    return original, data, idx, semantics["logits"]


def train_architectures(data, prior, logits, fit, prefix):
    """两种机制的对照：噪声通道最大似然与平方风险语义融合。

    每个阶段所有归一化、岭系数、分裂阈值、树叶只使用本阶段 fit 行。
    五星语义概率没有经过评分噪声逆变换，避免错误地放大其极端概率。
    """
    eps = float(data["epsilon"])
    target = (data["y"]-3*eps)/(1-eps)
    base = prior["prediction"]
    semantics = semantic_features(logits, base)
    features = np.column_stack([prior["features"], base, semantics]).astype(np.float32)
    linear_state, initial = fit_linear(base, semantics, target, fit, ridge=500.)
    np.savez_compressed(OUT/f"{prefix}_linear.npz", **linear_state)
    began = time.perf_counter()
    booster = ResidualHistogramBoost(rounds=400, depth=3, learning_rate=.035,
        min_leaf=250, l2=400, seed=5434).fit(features[fit], target[fit], initial[fit])
    tree_seconds = time.perf_counter()-began
    write_json(OUT/f"{prefix}_booster.json", booster.to_dict())
    began = time.perf_counter()
    channel, report = fit_channel(features, logits, data["y"], fit, eps, ridge=50., iterations=250)
    report["seconds"] = time.perf_counter()-began
    np.savez_compressed(OUT/f"{prefix}_channel.npz", **channel)
    write_json(OUT/f"{prefix}_channel_fit.json", report)
    predictions = {"baseline": base, "linear": initial,
                   "channel": predict_channel(features, logits, channel)}
    # 无标签分布审计发现测试身份全部有历史；在看语义效果之前预设一个有历史专家。
    # 缺历史行仍走全体模型，专家训练/门控均不使用任何确认标签。
    known = known_history_mask(prior["features"])
    began = time.perf_counter()
    warm_channel, warm_report = fit_channel(features, logits, data["y"], fit & known, eps,
                                            ridge=50., iterations=250)
    warm_report["seconds"] = time.perf_counter()-began
    np.savez_compressed(OUT/f"{prefix}_channel_warm.npz", **warm_channel)
    write_json(OUT/f"{prefix}_channel_warm_fit.json", warm_report)
    predictions["channel_warm"] = np.where(known, predict_channel(features, logits, warm_channel), predictions["channel"])
    for rounds in [100, 200, 400]:
        predictions[f"tree{rounds}"] = booster.predict(features, initial, rounds)
    if prefix == "confirm":
        # 同目标函数的去语义消融：只解释贡献，不加入冻结后的候选选择列表。
        no_semantic = np.column_stack([prior["features"], base]).astype(np.float32)
        zero_offset = np.zeros_like(logits)
        state, ablation_report = fit_channel(no_semantic, zero_offset, data["y"], fit & known, eps,
                                             ridge=50., iterations=250)
        predictions["channel_without_semantics_warm"] = np.where(known,
            predict_channel(no_semantic, zero_offset, state), predictions["channel"])
        write_json(OUT/"confirm_without_semantics_fit.json", ablation_report)
    write_json(OUT/f"{prefix}_timing.json", {"tree_seconds": tree_seconds,
        "channel_seconds": report["seconds"], "warm_channel_seconds": warm_report["seconds"],
        "rows": int(fit.sum()), "warm_rows": int((fit & known).sum()), "features": features.shape[1]})
    return predictions


def chosen_prediction(predictions, choice):
    name = choice["architecture"]
    if name == "mixture":
        candidate = .5*predictions["channel"]+.5*predictions["tree400"]
    elif name == "mixture_warm":
        candidate = .5*predictions["channel_warm"]+.5*predictions["tree400"]
    else:
        candidate = predictions[name]
    return predictions["baseline"]+choice["alpha"]*(candidate-predictions["baseline"])


def metrics(prediction, data, mask, known):
    """同时保留全体风险与贴近测试身份支持条件的风险，防止选择口径被隐去。"""
    result = evaluate(prediction[mask], data["y"][mask], data["stars"][mask], float(data["epsilon"]))
    selected = mask & known
    result["all_identities_known_risk"] = adjusted_risk(np.clip(prediction[selected], 1, 5), data["y"][selected], float(data["epsilon"]))
    result["all_identities_known_rows"] = int(selected.sum())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["develop", "final"], required=True)
    args = parser.parse_args()
    if args.stage == "final" and (OUT/"model").exists():
        raise FileExistsError("最终包已经存在，禁止覆盖已有候选；新实验请使用新目录。")
    if args.stage == "develop" and (OUT/"frozen_selection.json").exists():
        raise FileExistsError("本轮方案已冻结，禁止根据已查看的确认结果重选。")
    original, data, idx, logits = load_development()
    eps = float(data["epsilon"])
    dev = data["groups"] % np.uint64(100) == 0
    check = dev & ((data["groups"]//np.uint64(100)) % np.uint64(5) == 0)
    confirm = ~dev
    if args.stage == "develop":
        cached = prior_predictions(original, "inner", OUT)
        prior = {key: value[idx] for key, value in cached.items()}
        known = known_history_mask(prior["features"])
        predictions = train_architectures(data, prior, logits, dev & ~check, "inner")
        rows = []
        for architecture in ["linear", "channel", "channel_warm", "tree100", "tree200", "tree400", "mixture", "mixture_warm"]:
            for alpha in [.5, 1.]:
                choice = {"architecture": architecture, "alpha": alpha}
                p = chosen_prediction(predictions, choice)
                rows.append({**choice, "metrics": metrics(p, data, check, known)})
        baseline = metrics(prior["prediction"], data, check, known)
        chosen = min(rows, key=lambda r: r["metrics"]["all_identities_known_risk"]["latent_mse_estimate"])
        write_json(OUT/"inner_comparison.json", {"baseline": baseline, "candidates": rows,
            "primary_selection_metric": "all_identities_known_risk", "defined_before_semantic_effect_evaluation": True})
        write_json(OUT/"frozen_selection.json", chosen)
        print("第五轮内部冻结", json.dumps(chosen, ensure_ascii=False), flush=True)
        # 确认前已写出冻结方案；本阶段不重新选择架构、轮数、权重或噪声率。
        cached = prior_predictions(original, "confirm", OUT)
        prior = {key: value[idx] for key, value in cached.items()}
        known = known_history_mask(prior["features"])
        predictions = train_architectures(data, prior, logits, dev, "confirm")
        p = chosen_prediction(predictions, chosen)
        result = {"inherited": metrics(prior["prediction"], data, confirm, known),
                  "candidate": metrics(p, data, confirm, known),
                  "configuration_frozen_before_confirmation": True,
                  "historical_confirmation_reuse_disclosed": True}
        write_json(OUT/"confirmation.json", result)
        # 只读消融：这些确认分数仅解释贡献，不反向用于更换已经冻结的候选。
        ablations = {key: metrics(value, data, confirm, known)
                     for key, value in predictions.items()}
        raw_semantic = np.exp(logits-logits.max(axis=1, keepdims=True))
        raw_semantic /= raw_semantic.sum(axis=1, keepdims=True)
        ablations["external_semantic_alone"] = evaluate((raw_semantic@np.arange(1.,6.))[confirm], data["y"][confirm], data["stars"][confirm], eps)
        write_json(OUT/"confirmation_ablations.json", {"not_used_to_reselect": True, "metrics": ablations})
        pd.DataFrame({"id": data["id"][confirm], "rating": data["y"][confirm],
            "baseline": prior["prediction"][confirm], "prediction": np.clip(p[confirm], 1, 5),
            "groups": data["groups"][confirm], "all_identities_known": known[confirm]}).to_csv(OUT/"confirmation_predictions.csv", index=False)
        print("第五轮冻结确认", json.dumps(result, ensure_ascii=False), flush=True)
        return
    chosen = json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    confirmation = json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
    if confirmation["candidate"]["all_identities_known_risk"]["latent_mse_estimate"] >= confirmation["inherited"]["all_identities_known_risk"]["latent_mse_estimate"]:
        raise RuntimeError("冻结确认未改善，不把该实验包装为推荐提交。")
    full = prior_predictions(original, "final", OUT)
    prior = {"prediction": full["prediction"][idx], "features": full["features"][idx]}
    train_architectures(data, prior, logits, np.ones(len(idx), dtype=bool), "final")
    test = dict(np.load(OUT/"semantic_test.npz"))
    expected = np.load(ROOT/"task2/outputs/round3_20261105/test.npz")["id"]
    if not np.array_equal(expected, test["id"]):
        raise ValueError("语义测试预测错位。")
    base = full["test_prediction"]
    semantic = semantic_features(test["logits"], base)
    x = np.column_stack([full["test_features"], base, semantic]).astype(np.float32)
    from .semantic_fusion import predict_linear
    linear = predict_linear(base, semantic, dict(np.load(OUT/"final_linear.npz")))
    tree = ResidualHistogramBoost.from_dict(json.loads((OUT/"final_booster.json").read_text(encoding="utf-8")))
    predictions = {"baseline": base, "linear": linear,
        "channel": predict_channel(x, test["logits"], dict(np.load(OUT/"final_channel.npz")))}
    predictions["channel_warm"] = np.where(known_history_mask(full["test_features"]),
        predict_channel(x, test["logits"], dict(np.load(OUT/"final_channel_warm.npz"))), predictions["channel"])
    for rounds in [100, 200, 400]:
        predictions[f"tree{rounds}"] = tree.predict(x, linear, rounds)
    p = np.clip(chosen_prediction(predictions, chosen), 1, 5)
    pd.DataFrame({"id": expected, "rating": p}).to_csv(OUT/"submission.csv", index=False, float_format="%.10f")
    # 新包只复制所选推理依赖；旧模型仍只读，外部权重的许可证和来源一并保存。
    package = OUT/"model"
    package.mkdir(exist_ok=False)
    for source, destination in [(ROOT/"task2/outputs/round4_20261105/model", package/"prior_model"),
                                (OUT/"external_model", package/"external_model")]:
        destination.mkdir()
        for file in source.rglob("*"):
            relative = file.relative_to(source)
            # 推理选择CPU量化权重，未用到的GPU文件不重复复制到提交模型包。
            if relative.as_posix() == "onnx/model_fp16.onnx":
                continue
            target = destination/relative
            if file.is_dir():
                target.mkdir(exist_ok=True)
            else:
                shutil.copy2(file, target)
    for name in ["linear.npz", "channel.npz", "channel_warm.npz", "booster.json"]:
        shutil.copy2(OUT/("final_"+name), package/name)
    provenance_path = package/"external_model/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["experimental_file_not_bundled"] = "onnx/model_fp16.onnx"
    provenance["files"] = [record for record in provenance["files"] if record["file"] != "onnx/model_fp16.onnx"]
    write_json(provenance_path, provenance)
    write_json(package/"model.json", {"schema_version": 1, "chosen": chosen,
        "semantic_model_is_external": True, "competition_external_model_rule": "not_yet_verified",
        "semantic_runtime": {"provider": "cpu", "max_length": 128, "dynamic_length": True, "batch_size": 16},
        "meta_training_rows": len(idx), "original_training_rows": 7684645,
        "primary_selection_metric": "all_identities_known_risk",
        "epsilon_training": eps, "prior_model": "prior_model", "external_model": "external_model"})


if __name__ == "__main__":
    main()
