"""分析明确星级标题与评分的一致性，只输出聚合统计，不修改标签。

标题可能表达个人习惯或反讽，因此不将不一致自动认定为错误标签。
用精确匹配的 One/Five Stars 等常见标题做诊断，避免宽泛情感规则。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import ROOT, CSV_OPTIONS, write_json


def main():
    names = {"one star": 1, "one stars": 1, "two stars": 2, "three stars": 3,
             "four stars": 4, "five stars": 5}
    table = np.zeros((5, 5), dtype=np.int64)
    fit_table = np.zeros((5, 5), dtype=np.int64)
    valid_table = np.zeros((5, 5), dtype=np.int64)
    groups = np.load(ROOT / "task2/audit_compatible/metadata.npz")["groups"]
    titles = []
    rows = 0
    for frame in pd.read_csv(ROOT / "comp-5434-20261-project-task-2/train.csv",
                             usecols=["title", "rating"], chunksize=200000, **CSV_OPTIONS):
        title_rating = frame.title.fillna("").str.strip().str.lower().map(names)
        matched = title_rating.notna()
        codes = title_rating.fillna(0).to_numpy(dtype=np.uint8)
        titles.append(codes)
        fit = groups[rows:rows+len(frame)] % np.uint64(100) >= 2
        for chosen, destination in [(matched & fit, fit_table), (matched & ~fit, valid_table)]:
            np.add.at(destination, (codes[chosen]-1, frame.rating[chosen].to_numpy(dtype=int)-1), 1)
        np.add.at(table, (title_rating[matched].to_numpy(dtype=int)-1,
                         frame.rating[matched].to_numpy(dtype=int)-1), 1)
        rows += len(frame)
    result = {"training_rows": rows, "explicit_star_title_rows": int(table.sum()),
              "title_star_vs_observed_rating": table.tolist(),
              "agreement": float(np.trace(table) / table.sum()),
              "note": "描述性一致性分析，不能自动将矛盾标题认定为错误标签。"}
    path = ROOT / "task2/outputs/optimization_20261105"
    path.mkdir(exist_ok=True)
    # 只用外层训练部分估计替换率；验证评分与代理星级不参与参数估计。
    agreement = float(np.trace(fit_table)/fit_table.sum())
    epsilon = (1-agreement)/(1-1/5)
    result.update(fit_table=fit_table.tolist(),valid_table=valid_table.tolist(),
                  fit_only_uniform_replacement_rate=epsilon,
                  attenuation=1-epsilon,offset=3*epsilon,
                  assumption="精确星级标题是代理信号；仅提出均匀随机替换假设，不能认定其为真实标签。")
    np.save(path / "explicit_title_stars.npy", np.concatenate(titles))
    write_json(path / "title_audit_split.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
