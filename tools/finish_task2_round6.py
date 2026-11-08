"""第六轮候选交付：验证原文件、CSV、独立推理、回归测试并生成中文报告。

不执行Kaggle上传，不修改已知线上成绩，不覆盖原模型包或提交文件。
"""
import csv
import json
import math
import shutil
import unittest
from pathlib import Path

from task2.audit import ROOT,sha256,write_json

OUT=ROOT/"task2/outputs/round6_20261108"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    if (OUT/"submission.csv").read_bytes()!=(OUT/"reproduced.csv").read_bytes():
        raise RuntimeError("第六轮独立原始CSV推理不一致。")
    old=ROOT/"submissions/20261105_task2_round5"
    if sha256(old/"task2_semantic_noise_fusion.csv")!=read(old/"manifest.json")["submission_sha256"]:
        raise RuntimeError("旧0.54604对应文件发生变化。")
    # 早期Task1/Task2已上传候选也只读核对，防止整理新版本时覆盖。
    for folder in ["20261105","20261105_round2"]:
        base=ROOT/"submissions"/folder
        for name,meta in read(base/"manifest.json")["files"].items():
            if sha256(base/name)!=meta["sha256"]:raise RuntimeError("旧候选发生变化。")
    for folder,name in [("20261105_task2_round3","task2_nonlinear_residual.csv"),
                        ("20261105_task2_round4","task2_robust_refinement.csv")]:
        base=ROOT/"submissions"/folder
        if sha256(base/name)!=read(base/"manifest.json")["submission_sha256"]:raise RuntimeError("旧Task2候选变化。")
    with (OUT/"submission.csv").open(encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
    with (ROOT/"comp-5434-20261-project-task-2/sampleSubmission.csv").open(encoding="utf-8",newline="") as stream:sample=list(csv.DictReader(stream))
    if len(rows)!=10000 or list(rows[0])!=["id","rating"] or [r["id"] for r in rows]!=[r["id"] for r in sample]:
        raise RuntimeError("新提交列名、记录数或id顺序非法。")
    if not all(math.isfinite(float(r["rating"])) and 1<=float(r["rating"])<=5 for r in rows):raise RuntimeError("新评分范围非法。")
    suite=unittest.defaultTestLoader.discover(str(ROOT/"task2/tests"))
    with (OUT/"regression_tests.log").open("w",encoding="utf-8") as stream:
        tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not tests.wasSuccessful():raise RuntimeError("第六轮完整回归测试未通过。")
    timing=read(OUT/"reproduced.timing.json")
    if timing["semantic_checkpoint_reused"]:raise RuntimeError("独立复现必须使用新的工作目录。")
    result=read(OUT/"confirmation.json");chosen=read(OUT/"frozen_selection.json")
    protocol=read(OUT/"protocol.json");uncertainty=read(OUT/"paired_uncertainty.json")
    model=read(OUT/"model/model.json");fit=read(OUT/"final_training.json")
    selected_description={k:v for k,v in chosen.items() if k!="metrics"}
    training_seconds=fit.get("seconds",0.) if chosen["family"]!="mlp" else fit["anchor"]["seconds"]+fit["mlp"]["seconds"]
    parent=result["parent"];candidate=result["candidate"];ablation=result["without_cls"]
    p=parent["noise_model_risk"]["latent_rmse_estimate"];n=candidate["noise_model_risk"]["latent_rmse_estimate"]
    if n>=p:raise RuntimeError("确认风险未改善，不创建推荐候选。")
    destination=ROOT/"submissions/20261108_task2_round6";destination.mkdir(exist_ok=False)
    target=destination/"task2_cls_adapted.csv";shutil.copy2(OUT/"submission.csv",target)
    # 模型包内也保留理解材料，并明确原README属于第三方导出模型卡。
    shutil.copy2(ROOT/"task2/docs/BERT原理与本项目答辩指南.md",OUT/"model/UNDERSTANDING.md")
    (OUT/"model/MODEL_USAGE.md").write_text(
        "# 第六轮完整模型包\n\n使用项目源码中的task2.embedding_predict从原始测试CSV预测。\n"
        "model.json给出当前预测头与冻结BERT指纹；support保留第五轮依赖；head.npz是本项目训练的参数。\n"
        "README.md、ORIGINAL_MODEL_CARD.md和provenance.json是第三方来源的历史模型卡/下载说明。\n"
        "本轮老师允许外部模型（用户转述），条件是理解模型；请阅读UNDERSTANDING.md及项目结果报告。\n"
        "BERT权重冻结，768维CLS之后的头经过本项目训练。当前候选尚未获得新的Kaggle分数。\n",encoding="utf-8")
    proof={"date":"2026-11-08","previous_known_kaggle_best_user_report":.54604,"new_kaggle_score":None,
        "teacher_permission_external_models_user_report":True,"condition":"understand pretrained model",
        "choice":chosen,"rows":len(rows),"id_order_verified":True,"all_old_submissions_unchanged":True,
        "independent_raw_csv_prediction_byte_identical":True,"tests_passed":tests.testsRun,
        "submission_sha256":sha256(target),"model_location":"task2/outputs/round6_20261108/model",
        "protocol_sha256":sha256(OUT/"protocol.json"),"weights_of_bert_updated":False,
        "model_files":{file.relative_to(OUT/"model").as_posix():sha256(file) for file in (OUT/"model").rglob("*") if file.is_file()},
        "source_code_sha256":{name:sha256(ROOT/"task2"/name) for name in ["embedding_prepare.py","embedding_infer.py",
            "embedding_features.py","embedding_heads.py","embedding_train.py","embedding_predict.py","embedding_uncertainty.py",
            "embedding_structure_ablation.py"]}}
    write_json(destination/"manifest.json",proof)
    text=["# Task2第六轮优化结果（2026-11-08）","","## 结论与线上边界","",
        "用户反馈第五轮线上RMSE=0.54604，当前已知最佳。本轮未上传Kaggle，不预填新线上成绩。",
        "用户转述老师允许外部预训练模型，条件是自己理解模型；理解材料见task2/docs/BERT原理与本项目答辩指南.md。",
        "本轮取最后一层768维CLS训练项目自己的预测头，BERT内部权重冻结。",
        f"最终选中：{selected_description}。其余路线仅作为开发对照，不能全部写成最终模型。","",
        "## 验证协议与比较","",
        f"只依赖历史支持条件选出{protocol['rows']:,}条，内部训练{protocol['inner_fit']:,}、内部检查{protocol['inner_check']:,}、冻结确认{protocol['confirmation']:,}条。",
        "测试用户、商品和父商品均有历史；原词模型和查找表在验证时排除原外层验证行。",
        "验证分组有历史选型复用，预训练语料重叠未知，不声称是全新独立测试。",
        "本轮有历史子集的量化批组成与第五轮全子集不同，基线和新头使用同一批logits，保证条件一致。",
        "测试输入保持旧批量策略，另外核对第五轮基线和旧0.54604对应CSV一致。","",
        "| 冻结确认指标 | 同输入第五轮基线 | 第六轮 | 同头移除CLS |","|---|---:|---:|---:|",
        f"| 原观测RMSE | {parent['observed_rmse']:.6f} | {candidate['observed_rmse']:.6f} | {ablation['observed_rmse']:.6f} |",
        f"| 噪声模型风险RMSE估计 | {p:.6f} | {n:.6f} | {ablation['noise_model_risk']['latent_rmse_estimate']:.6f} |",
        f"| 无显式星级标题子组风险 | {parent['non_explicit_title_risk']['latent_rmse_estimate']:.6f} | {candidate['non_explicit_title_risk']['latent_rmse_estimate']:.6f} | {ablation['non_explicit_title_risk']['latent_rmse_estimate']:.6f} |","",
        f"风险估计相对变化{(n/p-1)*100:+.2f}%；不是新Kaggle成绩。",
        f"新减旧组重采样条件95%区间：{uncertainty['parent']['conditional_95_interval']}。",
        f"新减去CLS消融的条件95%区间：{uncertainty['without_cls']['conditional_95_interval']}。",
        "这些区间依赖固定均匀噪声假设，不包含历史选型、噪声模型错误或外部数据重叠/分布变化。","",
        "## 实现和成本","",
        "输入为旧59维、第五轮评分1维、CLS768维，共828维。统计标准化只用相应训练行。",
        "岭回归采用分块正规方程；线性分类器用自主噪声通道解析梯度；MLP的tanh、dropout、反向传播、梯度裁剪和Adam均自主实现。",
        "BERT内部权重没有更新，没有把本轮说成768万样本的全参数微调。",
        "原Dask词/历史流程继续继承，当前头训练与神经推理为本地增量阶段。",
        f"编码器开发推理：{read(OUT/'development.timing.json')['inference_seconds']:.3f}秒；部分时段与测试特征推理并行，存在CPU资源竞争。",
        f"最终头训练总计{training_seconds:.3f}秒，逐轮记录与优化器状态见final_training.json。",
        f"10000条原始CSV完整独立推理：{timing['complete_inference_seconds']:.3f}秒，不含写提交文件后续的哈希检查。",
        "硬件：i5-11300H、约16GB内存；冻结ONNX推理使用CPU四线程，头的NumPy运算使用原数值环境。",
        "上述时间是本轮增量成本，不含继承词模型、历史统计和原缓存从零构建。","",
        "## 交付与复现","",
        f"{tests.testsRun}项完整回归测试通过，包括独立正规方程、噪声logits与完整MLP链式梯度有限差分、训练标签隔离。",
        "相同256条输入原logits逐值一致，图只增加输出；最终独立CSV与首次CSV逐字节一致。",
        "旧提交哈希不变，新10000条记录、id顺序、表头和有限1–5评分均通过。",
        "提交文件：submissions/20261108_task2_round6/task2_cls_adapted.csv。",
        "完整模型：task2/outputs/round6_20261108/model；命令见task2/README.md第六轮章节。",""]
    if (OUT/"linear_anchor_confirmation_read_only.json").exists():
        anchor=read(OUT/"linear_anchor_confirmation_read_only.json")["metrics"]
        text += ["## 只读结构解释", "",
            f"同一MLP内固定的CLS线性锚点风险估计{anchor['noise_model_risk']['latent_rmse_estimate']:.6f}；最终MLP为{n:.6f}。",
            "此对照不重新训练或选择模型，仅解释非线性残差的点估计变化，没有单独为这两者差异构造区间，不能据此声称MLP普遍更好。",""]
    (ROOT/"docs/Task2第六轮优化结果-20261108.md").write_text("\n".join(text),encoding="utf-8")
    print(json.dumps({k:v for k,v in proof.items() if k not in ["model_files","source_code_sha256","choice"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
