"""第五轮冻结确认的配对组重采样，不反馈给模型选择。"""
import json
import numpy as np
import pandas as pd
from .audit import ROOT, write_json


def main():
    out = ROOT/"task2/outputs/round5_20261105"
    frame = pd.read_csv(out/"confirmation_predictions.csv")
    data = np.load(ROOT/"task2/outputs/round3_20261105/development.npz")
    epsilon = float(data["epsilon"])
    # 以确认CSV中的精确十进制组哈希为分组单位；uint64避免丢失高位精度。
    groups = frame.groups.to_numpy(dtype=np.uint64)
    y = frame.rating.to_numpy()
    def risk(prediction):
        return ((prediction-y)**2-epsilon*(prediction*prediction-6*prediction+11))/(1-epsilon)
    old = risk(frame.baseline.to_numpy())
    new = risk(frame.prediction.to_numpy())
    unique, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse)
    old_sum = np.bincount(inverse, weights=old)
    new_sum = np.bincount(inverse, weights=new)
    rng = np.random.default_rng(5434)
    differences = []
    for _ in range(500):
        sample = rng.integers(0, len(unique), size=len(unique))
        n = count[sample].sum()
        a, b = old_sum[sample].sum()/n, new_sum[sample].sum()/n
        if min(a, b) > 0:
            differences.append(np.sqrt(b)-np.sqrt(a))
    result = {"replicates": len(differences), "group_count": len(unique),
        "point_difference": float(np.sqrt(new.mean())-np.sqrt(old.mean())),
        "conditional_95_interval": np.quantile(differences, [.025, .975]).tolist(),
        "assumptions": "fixed uniform replacement noise, fixed chosen model",
        "not_included": "noise-model error, repeated historical selection, external model data overlap/shift",
        "not_a_kaggle_improvement_guarantee": True}
    known = frame.all_identities_known.to_numpy(dtype=bool)
    unique, inverse = np.unique(groups[known], return_inverse=True)
    count = np.bincount(inverse)
    old_sum = np.bincount(inverse, weights=old[known])
    new_sum = np.bincount(inverse, weights=new[known])
    differences = []
    for _ in range(500):
        sample = rng.integers(0, len(unique), size=len(unique))
        n = count[sample].sum()
        a, b = old_sum[sample].sum()/n, new_sum[sample].sum()/n
        if min(a, b) > 0:
            differences.append(np.sqrt(b)-np.sqrt(a))
    result["all_identities_known"] = {"rows": int(known.sum()), "replicates": len(differences),
        "point_difference": float(np.sqrt(new[known].mean())-np.sqrt(old[known].mean())),
        "conditional_95_interval": np.quantile(differences, [.025, .975]).tolist()}
    write_json(out/"paired_uncertainty.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
