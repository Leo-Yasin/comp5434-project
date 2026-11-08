"""核对查找统计排除验证评分，并从已保存的开发特征重算最终残差参数。

这里重训的是新增低维组合器，不声称重复训练整个旧词模型。
"""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import ROOT,write_json
from .lookup_features import KEYS
from .lookup_model import fit_residual
from .lookup_context import clean_context
from .calibration import correct


def main():
    out=ROOT/"task2/outputs/round2_20261105"
    manifest=json.loads((out/"keys_manifest.json").read_text(encoding="utf-8"))
    expected={key:[0,0.] for key in KEYS}
    for part in manifest["parts"]:
        with np.load(part["path"]) as data:
            fit=data["groups"]%np.uint64(100)>=2;y=data["rating"]
            for key in KEYS:
                mask=fit & (data["valid_"+key] if "valid_"+key in data else True)
                expected[key][0]+=int(mask.sum());expected[key][1]+=float(y[mask].sum(dtype=np.float64))
    for key in KEYS:
        with np.load(out/"stats"/f"split.{key}.npz") as state:
            if int(state["counts"].sum())!=expected[key][0] or float(state["sums"].sum())!=expected[key][1]:
                raise RuntimeError("查找统计与排除全部验证行后的计数/评分和不符。")
    cache=json.loads((ROOT/"task2/cache/manifest.json").read_text(encoding="utf-8"))
    contexts=[]
    for p in cache["parts"]:
        with np.load(p["prefix"]+".meta.npz") as data:valid=data["groups"]%np.uint64(100)<2
        contexts.append(np.load(p["prefix"]+".split.numeric.npy")[valid])
    description=json.loads((out/"model_context/model.json").read_text(encoding="utf-8"));eps=description["epsilon_split"]
    context=clean_context(np.concatenate(contexts),eps)
    validation=pd.read_csv(ROOT/"task2/outputs/optimization_20261105/converged/hybrid/validation_predictions.csv")
    base=correct(validation.prediction.to_numpy(),eps)
    data=np.load(out/"lookup_values.npz")
    began=time.perf_counter()
    coefficient=fit_residual(base,data["validation_means"],data["validation_counts"],validation.rating.to_numpy(),eps,context=context)
    seconds=time.perf_counter()-began
    saved=np.load(out/"model_context/residual.npz")["coefficient"]
    if not np.array_equal(coefficient,saved):raise RuntimeError("低维组合器从零求解参数不一致。")
    proof={"split_lookup_counts_and_sums_exclude_all_original_validation":True,
           "expected_training_counts_and_sums":expected,
           "residual_from_scratch_coefficients_identical":True,"residual_fit_seconds":seconds,
           "includes_design_and_normal_equation_solve":True,
           "full_inherited_base_model_retrained":False}
    write_json(out/"training_verification.json",proof)
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
