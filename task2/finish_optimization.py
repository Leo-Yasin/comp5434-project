"""在全部数据上延续冻结的优化配置，导出原值和噪声校正两个独立候选。

旧模型和旧提交从不覆盖。校正包复制参数文件并显式记录假设，避免只手改 CSV
而没有可复现的模型推理路径。此命令不向 Kaggle 发送文件。
"""
import json
from pathlib import Path
import shutil,subprocess,sys
import numpy as np
import pandas as pd
from .audit import ROOT,write_json,sha256
from .calibration import estimate_rate
from .predict import predict_file


def main():
    out=ROOT/"task2/outputs/optimization_20261105"
    summary=json.loads((out/"converged/hybrid/summary.json").read_text(encoding="utf-8"))
    rounds=summary["best"]["iteration"]
    full=out/"full"
    subprocess.run([sys.executable,"-X","utf8","-m","task2.train","--mode","full","--variants","hybrid",
                    "--epochs",str(rounds),"--initial-weights",str(ROOT/"task2/outputs/final/hybrid/last_weights.npz"),
                    "--output",str(full)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,"-X","utf8","-m","task2.export","--run",str(full)],cwd=ROOT,check=True)
    # 原模型包保持无校正，另建模型包。每个文件明确复制，避免修改共享基线。
    calibrated=full/"model_calibrated";calibrated.mkdir(exist_ok=False)
    for source in (full/"model").iterdir():
        if source.is_file():shutil.copy2(source,calibrated/source.name)
    diagnosis=json.loads((out/"title_audit_split.json").read_text(encoding="utf-8"))
    epsilon=estimate_rate(diagnosis["title_star_vs_observed_rating"])
    description=json.loads((calibrated/"model.json").read_text(encoding="utf-8"))
    description["score_calibration"]={"epsilon":epsilon,"uniform_mean":3.0,
        "method":"inverse symmetric uniform-label replacement expectation",
        "estimated_from":"full training exact star-title cross-table, no test labels",
        "assumption":"标题是代理信号；噪声模型是统计假设，线上效果尚待验证。"}
    write_json(calibrated/"model.json",description)
    test=ROOT/"comp-5434-20261-project-task-2/test.csv"
    raw_timing=predict_file(full/"model",test,out/"submission_optimized_raw.csv")
    cal_timing=predict_file(calibrated,test,out/"submission_optimized_calibrated.csv")
    predict_file(calibrated,test,out/"reproduced_calibrated.csv")
    if (out/"submission_optimized_calibrated.csv").read_bytes()!=(out/"reproduced_calibrated.csv").read_bytes():
        raise RuntimeError("独立校正模型重复推理 CSV 不一致。")
    sample=pd.read_csv(ROOT/"comp-5434-20261-project-task-2/sampleSubmission.csv")
    for file in ["submission_optimized_raw.csv","submission_optimized_calibrated.csv"]:
        p=pd.read_csv(out/file)
        if not p.id.equals(sample.id) or not np.isfinite(p.rating).all() or not p.rating.between(1,5).all():
            raise RuntimeError("提交 id、范围或数值非法。")
    write_json(out/"verification.json",{"status":"passed","new_full_iterations":rounds,
        "initial_full_training_iterations":12,"warm_start_restarts_search_directions":True,
        "duplicate_calibrated_prediction_byte_identical":True,"epsilon_full_train":epsilon,
        "raw_timing":raw_timing,"calibrated_timing":cal_timing,
        "kaggle_baseline_user_report":0.78159,"new_kaggle_score":None,
        "submissions":{file:sha256(out/file) for file in ["submission_optimized_raw.csv","submission_optimized_calibrated.csv"]}})


if __name__=="__main__":main()
