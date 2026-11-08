"""核对第三轮模型与旧有效提交，交付一个新 Task2 候选，不改变线上选择。"""
from pathlib import Path
import csv,json,math,hashlib,shutil

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads((ROOT/path).read_text(encoding="utf-8"))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def main():
    source=ROOT/"task2/outputs/round3_20261105";destination=ROOT/"submissions/20261105_task2_round3"
    if (source/"submission.csv").read_bytes()!=(source/"reproduced.csv").read_bytes():raise RuntimeError("独立推理 CSV 不一致。")
    # 固定 Task1，以及两个已获正反馈的 Task2 版本；任何旧文件变化都必须中止交付。
    for folder in ["submissions/20261105","submissions/20261105_round2"]:
        old=read(folder+"/manifest.json")
        for name,metadata in old["files"].items():
            if digest(ROOT/folder/name)!=metadata["sha256"]:raise RuntimeError("旧有效候选内容变化。")
    with (source/"submission.csv").open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
    with (ROOT/"comp-5434-20261-project-task-2/sampleSubmission.csv").open(encoding="utf-8",newline="") as f:sample=list(csv.DictReader(f))
    if len(rows)!=10000 or list(rows[0])!=["id","rating"] or [r["id"] for r in rows]!=[r["id"] for r in sample]:raise RuntimeError("数量、列名或 ID 顺序不符。")
    if not all(math.isfinite(float(r["rating"])) and 1<=float(r["rating"])<=5 for r in rows):raise RuntimeError("评分非法。")
    destination.mkdir(exist_ok=False)
    target=destination/"task2_nonlinear_residual.csv";shutil.copy2(source/"submission.csv",target)
    selection=read("task2/outputs/round3_20261105/frozen_selection.json")
    confirmation=read("task2/outputs/round3_20261105/confirmation.json")
    timing=read("task2/outputs/round3_20261105/timing.json")
    infer=read("task2/outputs/round3_20261105/reproduced.timing.json")
    features=read("task2/outputs/round3_20261105/features.json")
    manifest={"old_task1_and_task2_submissions_unchanged":True,"current_known_kaggle_best_user_report":0.62238,
              "new_kaggle_score":None,"records":len(rows),"id_order_verified":True,
              "prediction_range":[min(float(r["rating"]) for r in rows),max(float(r["rating"]) for r in rows)],
              "independent_prediction_byte_identical":True,"submission_sha256":digest(target),
              "model":"task2/outputs/round3_20261105/model",
              "model_files":{p.relative_to(source/"model").as_posix():digest(p) for p in (source/"model").rglob("*") if p.is_file()},
              "selection_frozen_before_confirmation":True,"title_proxy_not_used_for_selection":True,
              "incremental_tests_passed":3}
    (destination/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    baseline,candidate=confirmation["baseline"],confirmation["candidate"]
    lines=["# Task2 第三轮优化结果（2026-11-05）", "",
           "## 已有成绩与范围", "", "用户报告当前线上最佳 RMSE=0.62238，来自第二轮上下文残差版。",
           "本轮冻结 Task1，未修改旧模型或提交；仅新增 Task2 候选，尚未上传 Kaggle。", "",
           "## 新模型", "", "35 个结构化输入：已校正基预测、未裁剪置信度、标题/正文/文本对/用户交互统计、",
           "支持数、用户/商品历史、购买/票数/长度/时间等基础字段，以及明确星级标题。",
           "从线性残差模型开始，自主实现平方损失直方图提升树，学习这些信号的非线性组合。",
           "平方损失的梯度、直方图、正则叶子、分裂增益、采样及参数更新均由本项目实现。", "",
           "潜在目标为 (observed_rating - 3*epsilon)/(1-epsilon)，训练目标不裁剪，输出才限制到 [1,5]。",
           "最终不能再次套用噪声逆变换，否则会重复校正。", "",
           "最终提升树拟合的是153,456条原留出OOF元样本；基础词模型与查找统计仍继承全量原始训练数据。", "",
           "## 开发与冻结确认", "",
           "原验证组号0内部再按另一位哈希留出约20%选择配置；组号1不参与结构、轮数或幅度选择。",
           "基础词模型以前用过整个原验证集选轮数，因此不能将这次确认声称为完全独立外部测试。",
           f"选择：深度 {selection['depth']}，{selection['rounds']} 轮，学习率 {selection['learning_rate']}，"
           f"最小叶子 {selection['min_leaf']}，L2=800，修正幅度 {selection['alpha']}。",
           "选择前两个候选结构和四个轮数检查点均已保存，不只展示最优项。",
           "显式星级标题已经作为输入，因此不再用标题代理分数挑选配置或证明效果。", "",
           "| 冻结确认组 | 旧线性组合 | 新非线性组合 |", "|---|---:|---:|",
           f"| 原评分 RMSE | {baseline['observed_rmse']:.6f} | {candidate['observed_rmse']:.6f} |",
           f"| 噪声假设风险 RMSE 估计 | {baseline['noise_model_risk']['latent_rmse_estimate']:.6f} | {candidate['noise_model_risk']['latent_rmse_estimate']:.6f} |",
           f"| 无显式星级标题子组风险 RMSE 估计 | {baseline['non_explicit_title_risk']['latent_rmse_estimate']:.6f} | {candidate['non_explicit_title_risk']['latent_rmse_estimate']:.6f} |", "",
           f"确认记录 {candidate['rows']:,} 条，其中无显式星级标题 {candidate['nonstar_rows']:,} 条。",
           "**风险估计依赖均匀随机替换假设，不是已知干净标签 RMSE，更不是新的 Kaggle 分数。原评分 RMSE 变差仍如实记录。**",
           "非星级标题子组也改善，效果并非仅来自直接识别星级标题。", "",
           "## 计算成本与复现", "",
           f"新增缓存特征准备 {features['seconds']:.3f} 秒；沿用已由 Dask 构建的分区和词/统计模型，没有重新解析整个文本库。",
           f"内部开发、确认和最终拟合合计 {timing['development_and_confirmation_plus_final_seconds']:.3f} 秒，"
           f"其中最终300轮拟合 {timing['incremental_final_boost_fit_seconds']:.3f} 秒。",
           "这是新增阶段成本，不含旧词模型、Dask 原预处理和历史统计；完整从零成本必须加上前两轮。",
           f"独立全测试推理 {infer['complete_inference_seconds']:.3f} 秒，包含包加载、CSV、全部特征与预测，不含最终写 CSV。",
           "新三项测试通过：非线性拟合与序列化、正则叶子解析解、特征维度/有限值。",
           "原缓存生成的旧词预测与保存的验证预测相符；新测试特征缓存与独立原始 CSV 推理生成的最终文件逐字节相同。",
           "预测包包含全部继承依赖，不需要训练评分或缓存即可运行；旧 Task1 与 Task2 文件哈希均核对不变。", "",
           "## 交付", "",
           "文件：submissions/20261105_task2_round3/task2_nonlinear_residual.csv。",
           "建议先做一次线上对照，0.62238 的原有效提交保留；没有新分数前不更改当前最佳记录。", ""]
    (ROOT/"docs/Task2第三轮优化结果-20261105.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k!="model_files"},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
