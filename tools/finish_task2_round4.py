"""第四轮候选核对、不可变旧版本检查和中文结果报告，不执行线上提交。"""
from pathlib import Path
import csv,json,math,hashlib,shutil

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads((ROOT/path).read_text(encoding="utf-8"))


def digest(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):value.update(block)
    return value.hexdigest()


def main():
    source=ROOT/"task2/outputs/round4_20261105"
    if (source/"submission.csv").read_bytes()!=(source/"reproduced.csv").read_bytes():raise RuntimeError("独立预测不一致。")
    # 旧Task1、各轮Task2和0.60134文件全部只读核对。
    for folder in ["submissions/20261105","submissions/20261105_round2"]:
        manifest=read(folder+"/manifest.json")
        for name,meta in manifest["files"].items():
            if digest(ROOT/folder/name)!=meta["sha256"]:raise RuntimeError("旧候选内容变化。")
    prior=read("submissions/20261105_task2_round3/manifest.json")
    if digest(ROOT/"submissions/20261105_task2_round3/task2_nonlinear_residual.csv")!=prior["submission_sha256"]:
        raise RuntimeError("0.60134版本内容发生变化。")
    with (source/"submission.csv").open(encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
    with (ROOT/"comp-5434-20261-project-task-2/sampleSubmission.csv").open(encoding="utf-8",newline="") as stream:sample=list(csv.DictReader(stream))
    if len(rows)!=10000 or list(rows[0])!=["id","rating"] or [r["id"] for r in rows]!=[r["id"] for r in sample]:raise RuntimeError("提交数量、列名或顺序非法。")
    if not all(math.isfinite(float(r["rating"])) and 1<=float(r["rating"])<=5 for r in rows):raise RuntimeError("评分范围非法。")
    out=ROOT/"submissions/20261105_task2_round4";out.mkdir(exist_ok=False)
    target=out/"task2_robust_refinement.csv";shutil.copy2(source/"submission.csv",target)
    selected=read("task2/outputs/round4_20261105/frozen_selection.json")
    confirmation=read("task2/outputs/round4_20261105/confirmation.json")
    split=read("task2/outputs/round4_20261105/robust_split/summary.json")
    full=read("task2/outputs/round4_20261105/robust_full/summary.json")
    inner=read("task2/outputs/round4_20261105/inner_comparison.json")
    config=read("task2/outputs/round4_20261105/model/model.json")
    timing=read("task2/outputs/round4_20261105/reproduced.timing.json")
    uncertainty=read("task2/outputs/round4_20261105/paired_uncertainty.json")
    proof={"current_known_kaggle_best_user_report":0.60134,"new_kaggle_score":None,
           "task1_and_previous_task2_files_unchanged":True,"records":len(rows),"id_order_verified":True,
           "independent_prediction_byte_identical":True,"submission_sha256":digest(target),
           "tests_passed":18,"selection_frozen_before_confirmation":True,
           "model_location":"task2/outputs/round4_20261105/model",
           "model_files":{p.relative_to(source/"model").as_posix():digest(p) for p in (source/"model").rglob("*") if p.is_file()},
           "source_code_sha256":{name:digest(ROOT/"task2"/name) for name in ["robust_text.py","round4_features.py","round4_train.py","round4_predict.py"]}}
    (out/"manifest.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding="utf-8")
    old,new=confirmation["inherited"],confirmation["candidate"]
    without=min([r for r in inner["candidates"] if not r["use_auxiliary"]],key=lambda r:r["metrics"]["noise_model_risk"]["latent_mse_estimate"])
    lines=["# Task2 第四轮优化结果（2026-11-05）", "", "## 基线与边界", "",
           "用户报告第三轮线上 RMSE=0.60134，本轮保留该文件与所有旧候选，Task1不改动。",
           "新文件尚未上传，下面没有新的Kaggle分数。", "",
           "## 新算法：稳健辅助词信号", "",
           "参考固定的原词模型，按训练残差计算 a=min(1,1/abs(residual))，再用自主Dask-PCG解加权Ridge。",
           "这只是一轮固定Huber权重更新，不声称已经迭代到Huber损失的全局最优。",
           "验证行权重为零，计算训练梯度时不读取验证评分；权重隔离用验证标签NaN的单元测试检查。",
           f"开发训练 {split['training_rows']:,} 条、有效权重和 {split['effective_weight_sum']:.1f}；",
           f"全量训练 {full['training_rows']:,} 条、有效权重和 {full['effective_weight_sum']:.1f}，没有删除原始记录。",
           "稳健词模型的原始验证RMSE为1.201198，本身并非原评分指标上的优胜者，因此仅作为辅助输入。",
           "非线性组合同时看到原尺度、逆通道尺度和相对原词预测的差，不把辅助评分直接当成干净目标。", "",
           "## 开发对照与冻结确认", "",
           "沿用组号0内部选择、组号1确认；确认组在历史多轮已被查看，因此不是从未使用的外部测试。",
           f"仅继续残差的内部最佳风险估计 {without['metrics']['noise_model_risk']['latent_rmse_estimate']:.6f}；",
           f"加入辅助信号的内部最佳 {selected['metrics']['noise_model_risk']['latent_rmse_estimate']:.6f}，选择后冻结。",
           "选定：保留原300轮深度3模型，再新增200轮深度2、学习率0.04、叶子最少700、L2=800的残差树，幅度1。",
           "两阶段特征不同，不等同于从零拟合500棵同配置的树。", "",
           "| 确认组指标 | 0.60134方案对应模型 | 本轮候选 |", "|---|---:|---:|",
           f"| 原评分RMSE | {old['observed_rmse']:.6f} | {new['observed_rmse']:.6f} |",
           f"| 噪声假设下风险RMSE估计 | {old['noise_model_risk']['latent_rmse_estimate']:.6f} | {new['noise_model_risk']['latent_rmse_estimate']:.6f} |",
           f"| 无显式星级标题子组风险估计 | {old['non_explicit_title_risk']['latent_rmse_estimate']:.6f} | {new['non_explicit_title_risk']['latent_rmse_estimate']:.6f} |", "",
           "风险仍依赖均匀标签替换假设，不是已知干净标签实测，也不是Kaggle分数。星级代理未用于选择。",
           f"400次按评论组配对重采样：新减旧风险RMSE估计差 {uncertainty['point_new_minus_old_rmse_estimate']:.6f}，",
           f"条件95%区间 {uncertainty['conditional_percentile_interval_95']}。",
           "该区间只描述固定模型/假设下抽样波动，不计噪声假设错误、历史选型偏差或线上分布变化，不能保证线上提升。", "",
           "## 效率、硬件和复现", "",
           "单机i5-11300H、约15.74GiB内存、8逻辑CPU，辅助词模型实际Dask三工作进程、各一线程，无GPU。",
           f"全量训练工作进程PID：{full['actual_worker_pids']}。",
           f"辅助模型12次全数据加权算子迭代，总拟合 {full['total_fit_seconds']:.3f} 秒，平均每次 {full['mean_iteration_seconds']:.3f} 秒。",
           f"新增最终残差拟合 {config['incremental_fit_seconds']:.3f} 秒；独立10000条推理 {timing['complete_inference_seconds']:.3f} 秒。",
           "这些是增量阶段成本，不包含继承模型和原缓存从零构建成本，也不含Dask启动或CSV写盘。",
           "PCG迭代每次遍历全训练分区；新增树的轮数不与PCG迭代混称相同epoch。详细逐轮时间保留。",
           "18项回归测试通过，包括加权正规方程对独立直接解、验证权重隔离、原梯度/噪声校正/新提升树检查。",
           "独立进程只用原始test.csv和包内依赖重新预测，与首次文件逐字节一致。",
           "所有旧提交哈希未变，新文件10000条、id顺序、列名与1–5范围均检查通过。", "",
           "## 交付", "",
           "submissions/20261105_task2_round4/task2_robust_refinement.csv。",
           "模型包 task2/outputs/round4_20261105/model/，不依赖训练缓存即可预测。",
           "建议先测一次新候选，与0.60134对照，取得实际分数再决定最终保留。", ""]
    (ROOT/"docs/Task2第四轮优化结果-20261105.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({k:v for k,v in proof.items() if k not in ["model_files","source_code_sha256"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
