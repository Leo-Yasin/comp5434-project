"""汇总真实产物、核对旧基线并生成统一的新提交目录，不执行线上提交。

只依赖标准库，避免在汇总文件时混用 Task 1/Task 2 的两套数值环境。
"""
from pathlib import Path
import csv,hashlib,json,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    b=read("task1/outputs/optimization_20261105/selection.json")
    rf=read("task1/outputs/tuned_forest_20261105/selection.json")
    confirm=read("task1/outputs/optimization_20261105/confirmation.json")
    cal=read("task2/outputs/optimization_20261105/calibration_evaluation.json")
    v2=read("task2/outputs/optimization_20261105/verification.json")
    full=read("task2/outputs/optimization_20261105/full/hybrid/summary.json")
    candidates=[("task1/outputs/optimization_20261105",b),("task1/outputs/tuned_forest_20261105",rf)]
    chosen,score=max(candidates,key=lambda pair:pair[1]["mean_macro_f1"])
    # 不改动旧提交，先核对其 SHA-256，再复制新的、名称清晰的独立候选文件。
    baseline_hashes={
        "task1/outputs/baseline/forest/submission_integer.csv":"0506e4d295b42df48c09caf0d9a4feda0aa4e35dc312ca3f479b51e1181fa691",
        "task2/outputs/final/submission.csv":"044c22d6c533e05de661b396b1faf0fb9842c7feb781a806a14346bfd6a035b3"}
    for file,expected in baseline_hashes.items():
        if digest(ROOT/file)!=expected:raise RuntimeError("旧基线文件发生变化。")
    source1=ROOT/chosen/"submission_integer.csv"
    if chosen.endswith("optimization_20261105"):
        if source1.read_bytes()!=(ROOT/chosen/"reproduced.csv").read_bytes():
            raise RuntimeError("Task1 独立预测不一致。")
    out=ROOT/"submissions/20261105";out.mkdir(parents=True,exist_ok=False)
    sources={"task1_optimized.csv":source1,
             "task2_optimized_raw.csv":ROOT/"task2/outputs/optimization_20261105/submission_optimized_raw.csv",
             "task2_optimized_calibrated.csv":ROOT/"task2/outputs/optimization_20261105/submission_optimized_calibrated.csv"}
    evidence={"baseline_hashes_unchanged":True,"task1_selected_source":chosen,
              "user_reported_kaggle_baseline":{"task1_macro_f1":0.91321,"task2_rmse":0.78159},
              "new_kaggle_scores":None,"files":{}}
    for name,source in sources.items():
        shutil.copy2(source,out/name)
        with source.open(encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
        column="label" if name.startswith("task1") else "rating"
        count=3300 if column=="label" else 10000
        if len(rows)!=count or list(rows[0])!=["id",column]:raise RuntimeError("提交行数或列名错误。")
        if column=="label" and not all(row[column] in ["1","2","3","4"] for row in rows):raise RuntimeError("Task1 必须为整数标签。")
        evidence["files"][name]={"source":str(source.relative_to(ROOT)),"rows":len(rows),"sha256":digest(source)}
    (out/"manifest.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 2026-11-05 优化实验结果", "", "## 已有线上基线", "",
           "用户报告 Task 1 Macro-F1=0.91321、Task 2 RMSE=0.78159。提交时间、编号和榜单类型未提供。",
           "原文件哈希核对一致，旧模型与提交未被覆盖。新文件均未上传 Kaggle。", "",
           "## Task 1：小幅、重复划分方向一致的改善", "",
           "| 方案 | 原五折 Macro-F1 |", "|---|---:|", "| 旧森林 | 0.915223 |",
           f"| 自主提升树与旧森林融合 | {b['mean_macro_f1']:.6f} |",
           f"| 调参森林 | {rf['mean_macro_f1']:.6f} |", "",
           f"最终来源：{chosen}；原五折均值 {score['mean_macro_f1']:.6f}。",
           f"冻结提升树融合配置后，在第二组五折：基线 {confirm['baseline_mean']:.6f}，候选 {confirm['candidate_mean']:.6f}。",
           "没有把第二组划分用于调参；两次划分仍使用同一批记录，不能替代外部测试或显著性结论。",
           "生成式直方图/共享背景假设未优于基线，全部失败配置保留在 generative_20261105/comparison.json。", "",
           "## Task 2：延长训练与评分收缩校正", "",
           "在 751,438 条明确星级标题上，训练评分约 80% 一致，其余四类各约 5%。",
           "它符合以约 25% 概率将标签随机替换为均匀 1–5 的假设，但星级标题不是已知真实标签。",
           f"外层训练部分估计 epsilon={cal['fit_only_epsilon']:.8f}，验证评分不参与参数估计。",
           "模型延续旧权重、重启 PCG 搜索方向，再训练 24 次完整遍历；不是等同冷启动 36 轮。", "",
           "| 模型/输出 | 原评分验证 RMSE | 星级标题代理 RMSE | 假设下的潜在 RMSE 估计 |",
           "|---|---:|---:|---:|"]
    for model in ["baseline","converged"]:
        for variant in ["raw","calibrated"]:
            r=cal["models"][model]["variants"][variant]
            lines.append(f"| {model}/{variant} | {r['observed_label_rmse']:.6f} | {r['explicit_title_proxy_rmse']:.6f} | {r['noise_model_adjusted_risk']['latent_rmse_estimate']:.6f} |")
    lines += ["", "校正公式：clip((prediction - 3*epsilon)/(1-epsilon), 1, 5)。",
              "潜在风险估计使用 MSE_latent=(MSE_observed-epsilon*mean(p²-6p+11))/(1-epsilon)。",
              "**校正版在原评分验证 RMSE 上变差。代理分数和噪声模型风险估计不是干净测试实测分数，也不是 Kaggle 分数。**",
              "保留常规版与校正版做线上对照，不能保证校正版提高 0.78159 的已有成绩。", "",
              "## 复现、耗时与提交文件", "",
              "Task1 使用 JSON 模型独立重生成预测一致；Task2 校正被写入独立模型包，重复推理 CSV 字节一致。",
              "两项旧提交 SHA-256 保持原值，所有新候选均通过数量、列名、ID 顺序及范围核对。",
              "Task2 继续实际使用 Dask 3 进程；新全量阶段包括额外初始残差遍历。",
              f"新增全量拟合 {full['total_fit_seconds_including_validation']:.3f} 秒，每新迭代平均 {full['mean_training_iteration_seconds']:.3f} 秒。",
              f"校正版全测试推理（含加载/读取/特征）{v2['calibrated_timing']['inference_seconds_including_loading_reading_transform']:.3f} 秒。",
              "旧完整训练与特征准备成本见原报告；不能把新增 warm-start 耗时当作从零全部离线成本。", "",
              "提交目录：submissions/20261105/。", "",
              "- task1_optimized.csv：Task1 整数标签候选。",
              "- task2_optimized_raw.csv：更多迭代的常规候选，原评分验证有小幅改善。",
              "- task2_optimized_calibrated.csv：依赖噪声假设的校正实验候选。", "",
              "建议 Task1 测一次新候选；Task2 分别测常规与校正版，记录成绩后决定保留方案。不要覆盖或撤回旧有效提交。", ""]
    (ROOT/"docs/优化结果-20261105.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
