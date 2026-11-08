"""汇总第二轮全部结果、校验独立预测，发布两个本地候选；不进行线上提交。"""
from pathlib import Path
import csv,json,hashlib,math,shutil

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads((ROOT/path).read_text(encoding="utf-8"))


def digest(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):value.update(block)
    return value.hexdigest()


def main():
    t1=ROOT/"task1/outputs/round2_20261105";t2=ROOT/"task2/outputs/round2_20261105"
    mix=read("task1/outputs/round2_20261105/selection.json")
    mix_check=read("task1/outputs/round2_20261105/confirmation.json")
    core=read("task1/outputs/core_expert_20261105/selection.json")
    core_check=read("task1/outputs/core_expert_20261105/confirmation.json")
    simple=read("task2/outputs/round2_20261105/evaluation.json")
    context=read("task2/outputs/round2_20261105/context_evaluation.json")
    keys=read("task2/outputs/round2_20261105/keys_manifest.json")
    refit=read("task2/outputs/round2_20261105/training_verification.json")
    timing=read("task2/outputs/round2_20261105/reproduced_context.timing.json")
    if (t1/"submission.csv").read_bytes()!=(t1/"reproduced.csv").read_bytes():raise RuntimeError("Task1 独立预测不一致。")
    if (t2/"submission_context.csv").read_bytes()!=(t2/"reproduced_context.csv").read_bytes():raise RuntimeError("Task2 独立预测不一致。")
    old=read("submissions/20261105/manifest.json")
    for name,state in old["files"].items():
        if digest(ROOT/"submissions/20261105"/name)!=state["sha256"]:raise RuntimeError("上一轮候选被修改。")
    out=ROOT/"submissions/20261105_round2";out.mkdir(exist_ok=False)
    files={"task1_experimental_mixture.csv":(t1/"submission.csv","comp-5434-20261-project-task-1/test.csv","label",3300),
           "task2_context_residual.csv":(t2/"submission_context.csv","comp-5434-20261-project-task-2/test.csv","rating",10000)}
    manifest={"prior_candidate_files_unchanged":True,"kaggle_task2_current_best_user_report":0.64452,
              "new_kaggle_scores":None,"task1_not_claimed_material_improvement":True,
              "tests_passed":{"task1":14,"task2":12},"files":{},"model_files":{}}
    for name,(source,test,column,n) in files.items():
        with source.open(encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
        with (ROOT/test).open(encoding="utf-8",newline="") as stream:expected=list(csv.DictReader(stream))
        if len(rows)!=n or list(rows[0])!=["id",column] or [r["id"] for r in rows]!=[r["id"] for r in expected]:raise RuntimeError("提交列名、顺序或行数错误。")
        if not all(math.isfinite(float(r[column])) and 1<=float(r[column])<=(4 if column=="label" else 5) for r in rows):raise RuntimeError("标签范围错误。")
        if column=="label" and not all(r[column] in ["1","2","3","4"] for r in rows):raise RuntimeError("Task1 必须用整数。")
        shutil.copy2(source,out/name)
        manifest["files"][name]={"source":source.relative_to(ROOT).as_posix(),"sha256":digest(source),
                                 "records":n,"id_order_verified":True,"independent_prediction_byte_identical":True}
    for directory in [t1/"package",t2/"model_context"]:
        manifest["model_files"][directory.relative_to(ROOT).as_posix()]={p.relative_to(directory).as_posix():digest(p)
            for p in directory.rglob("*") if p.is_file()}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 第二轮优化结果（2026-11-05）", "", "## 已知线上基线与本轮结论", "",
           "Task1 用户仅报告仍在 0.913 级别；Task2 上轮校正版的用户报告线上 RMSE=0.64452。",
           "本轮没有提交 Kaggle，没有新线上分数，原有效候选未修改。",
           "Task1 多条路线未找到有说服力的大幅改善；保留轻量混合模型实验版供一次对照。",
           "Task2 文本查找统计与历史上下文残差在开发确认中改善，生成新的独立候选。", "",
           "## Task1：模型比较与未改善结果", "", "| 方案 | 第一组五折 F1 | 第二组五折 F1 |",
           "|---|---:|---:|", "| 第一轮候选 | 0.916247 | 0.915553 |",
           f"| 多模态混合实验版 | {mix['mean_f1']:.6f} | {sum(r['candidate'] for r in mix_check)/5:.6f} |",
           f"| 支持区间专家 | {core['mean_f1']:.6f} | {core_check['candidate_mean']:.6f} |", "",
           "支持专家第一组较高，第二组未保持优势，不能只展示第一组最高分；本次不将其推荐为主候选。",
           "混合实验版每类 3 个高斯-类别模式，75% 第一轮预测 + 25% 新密度预测。",
           "随机阈值树和容量配额未优于既有方案，结果文件保留。配额算法用小规模穷举检验最优，",
           "但不把均衡测试先验当成事实，也不推荐其上传文件。",
           "各方案都在同一数据反复开发，第二组检查也不等同独立测试；Task1 当前只能说改善有限。", "",
           "## Task2：新增信息与隔离协议", "",
           "对全部 768 万条训练评论构建规范化标题、正文、文本对及用户/商品交互键。",
           f"Dask 实际使用 {keys['workers']} 个工作进程，观测到的 PID：{keys['actual_worker_pids']}。",
           "所有查找统计排除原来的全部外层验证评分，训练侧计数与评分和已经独立核对。",
           "组合器在组号 0/1 两半对称交叉拟合，固定 ridge=100 和收缩尺度 2/10/50。",
           "基础模型曾使用原验证集选择迭代，不能将新检查声称为完全独立外部测试。",
           "最终组合器在全部开发 OOF 特征上拟合，最终查找表才使用全部训练标签；测试标签未使用。", "",
           "| 方案 | 原评分验证 RMSE | 星级标题代理 RMSE | 噪声假设下风险 RMSE 估计 |",
           "|---|---:|---:|---:|"]
    rows=[("上一轮校正版",simple["overall_cross_fitted"]["baseline"]),
          ("查找统计残差",simple["overall_cross_fitted"]["candidate"]),
          ("查找 + 历史上下文",context["overall"])]
    for name,row in rows:
        lines.append(f"| {name} | {row['observed_rmse']:.6f} | {row['explicit_title_proxy_rmse']:.6f} | {row['noise_model_adjusted_risk']['latent_rmse_estimate']:.6f} |")
    lines += ["", "上下文方案在两半风险估计均改善：0.683566→0.660119、0.685968→0.664583。",
              "**原评分 RMSE 变差。标题代理与风险估计不是干净标签实测，不是新的 Kaggle 分数。**",
              "噪声校正的均匀替换假设延续上一轮；它在已知线上结果有效，不代表所有新修改必然有效。", "",
              "## 覆盖率与效率", "", "| 新统计键 | 测试记录中有历史的比例 |", "|---|---:|"]
    for key,value in simple["coverage"].items():lines.append(f"| {key} | {value['test_seen_fraction']:.2%} |")
    lines += ["", "同用户同商品为零覆盖，没有将这一项包装成性能贡献。",
              f"新增 Dask 键准备 {keys['seconds']:.3f} 秒；分字段查找统计归约合计 {sum(simple['statistics_seconds'].values()):.3f} 秒。",
              f"组合器从零重算参数耗时 {refit['residual_fit_seconds']:.3f} 秒，包括特征设计和正规方程求解，系数与保存值完全一致。",
              "组合器是 28 维一次线性方程求解，不使用 epoch；继承基模型的逐轮与完整训练成本保留在上一轮报告。",
              "上述是增量操作成本，不能将它们声称为从零构建整个系统的总离线时间。",
              f"独立完整推理（加载两级模型、读 CSV、全部特征和预测）{timing['complete_inference_seconds']:.3f} 秒。",
              "硬件/包版本保存在 task2/outputs/round2_20261105/environment.json；新推理不需要训练缓存。", "",
              "## 复现与文件", "",
              "Task1 14 项、Task2 12 项测试通过。两个新候选独立推理的 CSV 都逐字节一致。",
              "三份上一轮提交的 SHA-256 全部保持不变。两个新文件的列名、ID 顺序、范围和数量检查通过。", "",
              "- submissions/20261105_round2/task1_experimental_mixture.csv：保守实验候选，未证明明显提升。",
              "- submissions/20261105_round2/task2_context_residual.csv：本轮 Task2 候选。",
              "- submissions/20261105_round2/manifest.json：模型/预测哈希和核对记录。", "",
              "Task1 建议保留旧有效提交，最多先测一次新实验版；Task2 新候选与 0.64452 版本对照后再决定最终选择。", ""]
    (ROOT/"docs/第二轮优化结果-20261105.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"files":manifest["files"],"tests_passed":manifest["tests_passed"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
