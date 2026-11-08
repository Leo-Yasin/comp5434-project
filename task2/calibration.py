"""在可检验的均匀标签替换假设下校正评分收缩。

若以 epsilon 概率将原评分替换为均匀 1–5，则 E[Y_observed|x]
= (1-epsilon)*E[Y_latent|x] + 3*epsilon。因此校正为 (p-3*epsilon)/(1-epsilon)。
参数仅用训练内精确星级标题统计估计；标题是代理信号，不宣称它是测试真值。
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import ROOT,write_json


def estimate_rate(table):
    table=np.asarray(table,dtype=float)
    if table.shape!=(5,5) or table.sum()<=0:
        raise ValueError("需要非空的 5x5 标题-评分计数。")
    return float(np.clip((1-np.trace(table)/table.sum())/0.8,0,0.6))


def correct(prediction,epsilon):
    if not 0<=epsilon<1:
        raise ValueError("替换率必须在 [0,1) 内。")
    return np.clip((np.asarray(prediction)-3*epsilon)/(1-epsilon),1,5)


def adjusted_risk(prediction,observed,epsilon):
    """给定均匀替换假设，从含噪验证 MSE 估计潜在干净评分风险。

    MSE_observed=(1-epsilon)*MSE_latent + epsilon*E[(prediction-U)^2]，
    U 均匀取 1..5，E[U]=3、E[U²]=11。负估计保留而不伪装成零误差。
    这是假设依赖的估计，不等于已知干净标签上的实测 RMSE。
    """
    p=np.asarray(prediction);observed=np.asarray(observed)
    mse=float((np.mean((p-observed)**2)-epsilon*np.mean(p*p-6*p+11))/(1-epsilon))
    return {"latent_mse_estimate":mse,"latent_rmse_estimate":float(np.sqrt(mse)) if mse>=0 else None}


def evaluate_predictions(csv,stars,epsilon):
    """原评分 RMSE 与标题代理 RMSE 分开记录，不能将代理分数称为真实测试分数。"""
    frame=pd.read_csv(csv);proxy=stars[frame.id.to_numpy(dtype=int)]
    raw=frame.prediction.to_numpy();calibrated=correct(raw,epsilon)
    result={"rows":len(frame),"proxy_rows":int((proxy>0).sum()),"variants":{}}
    for key,p in [("raw",raw),("calibrated",calibrated)]:
        result["variants"][key]={
            "observed_label_rmse":float(np.sqrt(np.mean((p-frame.rating.to_numpy())**2))),
            "explicit_title_proxy_rmse":float(np.sqrt(np.mean((p[proxy>0]-proxy[proxy>0])**2))),
            "noise_model_adjusted_risk":adjusted_risk(p,frame.rating.to_numpy(),epsilon),
            "note":"标题为代理，校正风险依赖均匀替换假设；都不是已知干净测试真值。"}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--include-converged",action="store_true")
    args=parser.parse_args()
    out=ROOT/"task2/outputs/optimization_20261105"
    diagnosis=json.loads((out/"title_audit_split.json").read_text(encoding="utf-8"))
    stars=np.load(out/"explicit_title_stars.npy")
    epsilon=estimate_rate(diagnosis["fit_table"])
    results={"fit_only_epsilon":epsilon,"full_training_epsilon":estimate_rate(diagnosis["title_star_vs_observed_rating"]),
             "parameter_source":"只用训练评分和精确标题代理，不使用测试评分或 Kaggle 分数拟合。",
             "models":{"baseline":evaluate_predictions(ROOT/"task2/outputs/development/hybrid/validation_predictions.csv",stars,epsilon)}}
    if args.include_converged:
        results["models"]["converged"]=evaluate_predictions(out/"converged/hybrid/validation_predictions.csv",stars,epsilon)
    write_json(out/"calibration_evaluation.json",results)
    # 原基线的校正版作为独立、低成本候选，主模型保持原始权重不被修改。
    prediction=pd.read_csv(ROOT/"task2/outputs/final/submission.csv")
    prediction.rating=correct(prediction.rating.to_numpy(),results["full_training_epsilon"])
    prediction.to_csv(out/"submission_baseline_calibrated.csv",index=False,float_format="%.10f")
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
