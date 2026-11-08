"""扩展审计：消融提示通道模型更优，先登记两候选，再看本轮未使用的哈希区间。

此区间仍属于历史外层验证，不称为从未被任何历史模型使用的独立测试。
不重训、不改参数、不扫描权重；仅比较已保存的confirm通道与400轮树。
"""
import argparse
import datetime
import json
import numpy as np
import pandas as pd

from .audit import ROOT, write_json, sha256
from .semantic_fusion import semantic_features, known_history_mask, predict_linear
from .semantic_train import metrics
from .noise_softmax import predict_channel
from .regression_boost import ResidualHistogramBoost

OUT = ROOT/"task2/outputs/round5_20261105"


def prepare():
    protocol = OUT/"extra_frozen_protocol.json"
    if protocol.exists():
        raise FileExistsError("扩展审计已登记，禁止重新选择区间。")
    data = np.load(ROOT/"task2/outputs/round3_20261105/development.npz")
    g = data["groups"]
    index = np.flatnonzero((g % np.uint64(100) == 1) & ((g//np.uint64(500)) % np.uint64(8) == 3))
    text = pd.read_csv(OUT/"validation_text.csv", keep_default_na=False)
    if not np.array_equal(text.id.to_numpy(), data["id"]):
        raise ValueError("额外审计文本id错位。")
    text.iloc[index].to_csv(OUT/"extra_text.csv", index=False)
    np.savez_compressed(OUT/"extra_selection.npz", index=index, id=data["id"][index])
    write_json(protocol, {"registered_at": datetime.datetime.now().astimezone().isoformat(),
        "reason": "first confirmation read-only ablation favored channel; require new current-round audit",
        "selection_rule": "groups % 100 == 1 and (groups // 500) % 8 == 3",
        "rows": len(index), "extra_semantic_effect_not_evaluated": True,
        "candidates": ["tree400 alpha1", "channel alpha1"],
        "primary_metric": "all_identities_known_risk", "refit_or_retuning_allowed": False,
        "history_reuse_disclosed": True,
        "model_hashes": {name: sha256(OUT/name) for name in ["confirm_linear.npz", "confirm_channel.npz", "confirm_booster.json"]}})
    print("扩展审计固定区间", len(index), flush=True)


def evaluate():
    protocol = json.loads((OUT/"extra_frozen_protocol.json").read_text(encoding="utf-8"))
    for name, expected in protocol["model_hashes"].items():
        if sha256(OUT/name) != expected:
            raise RuntimeError("扩展审计冻结参数已改变。")
    original = dict(np.load(ROOT/"task2/outputs/round3_20261105/development.npz"))
    ix = np.load(OUT/"extra_selection.npz")["index"]
    n = len(original["id"])
    data = {k: v[ix] if v.ndim and v.shape[0] == n else v for k, v in original.items()}
    semantic = np.load(OUT/"semantic_extra.npz")
    if not np.array_equal(data["id"], semantic["id"]):
        raise ValueError("扩展语义审计id错位。")
    cached = np.load(OUT/"prior_confirm.npz")
    base, old = cached["prediction"][ix], cached["features"][ix]
    known = known_history_mask(old)
    sf = semantic_features(semantic["logits"], base)
    x = np.column_stack([old, base, sf]).astype(np.float32)
    linear = predict_linear(base, sf, dict(np.load(OUT/"confirm_linear.npz")))
    tree = ResidualHistogramBoost.from_dict(json.loads((OUT/"confirm_booster.json").read_text(encoding="utf-8")))
    gb = np.clip(tree.predict(x, linear, 400), 1, 5)
    channel = predict_channel(x, semantic["logits"], dict(np.load(OUT/"confirm_channel.npz")))
    all_rows = np.ones(len(ix), dtype=bool)
    report = {"tree400": metrics(gb, data, all_rows, known),
              "channel": metrics(channel, data, all_rows, known),
              "baseline": metrics(base, data, all_rows, known)}
    eps = float(data["epsilon"]); y = data["y"]
    def risk(p):
        return ((p-y)**2-eps*(p*p-6*p+11))/(1-eps)
    a, b = risk(gb)[known], risk(channel)[known]
    unique, inverse = np.unique(data["groups"][known], return_inverse=True)
    count = np.bincount(inverse)
    old_sum, new_sum = np.bincount(inverse, weights=a), np.bincount(inverse, weights=b)
    rng = np.random.default_rng(5435); delta = []
    for _ in range(500):
        rows = rng.integers(0, len(unique), len(unique)); total = count[rows].sum()
        av, bv = old_sum[rows].sum()/total, new_sum[rows].sum()/total
        if min(av, bv) > 0:
            delta.append(np.sqrt(bv)-np.sqrt(av))
    interval = np.quantile(delta, [.025, .975]).tolist()
    report["channel_minus_tree_known"] = {"difference": float(np.sqrt(b.mean())-np.sqrt(a.mean())),
        "conditional_95_interval": interval, "not_accounted_for": "noise error, historical reuse, external data overlap/shift"}
    report["release_channel_as_additional_candidate"] = bool(b.mean() < a.mean() and interval[1] < 0)
    report["parameters_not_changed_after_extra_audit"] = True
    write_json(OUT/"extra_audit.json", report)
    if report["release_channel_as_additional_candidate"]:
        # 附加说明随包保存，部署时不需要找到本次训练目录；原model.json仍指向预选树。
        write_json(OUT/"model/channel_candidate.json", {
            "release_channel_as_additional_candidate": True,
            "architecture": "channel", "alpha": 1.,
            "final_channel_sha256": sha256(OUT/"model/channel.npz"),
            "extra_audit_sha256": sha256(OUT/"extra_audit.json"),
            "audit": report,
            "competition_external_model_rule": "not_yet_verified"})
    pd.DataFrame({"id": data["id"], "rating": y, "baseline": base,
        "tree400": gb, "channel": channel, "known": known}).to_csv(OUT/"extra_audit_predictions.csv", index=False)
    print("扩展审计", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "evaluate"], required=True)
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else evaluate()
