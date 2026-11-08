"""第五轮交付核对、完整回归测试、不可变旧版本检查与中文结果报告。

只整理本地文件，不登录Kaggle或执行线上提交。外部模型规则状态始终明确保留。
"""
import csv
import hashlib
import json
import math
import shutil
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"task2/outputs/round5_20261105"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    protocol = read(SOURCE/"frozen_protocol.json")
    for name, expected in protocol["source_hashes"].items():
        if digest(ROOT/"task2"/name) != expected:
            raise RuntimeError("预先登记的模型源码发生变化，必须如实审计后再交付。")
    if (SOURCE/"submission.csv").read_bytes() != (SOURCE/"reproduced.csv").read_bytes():
        raise RuntimeError("完整独立推理与首次文件不一致。")
    for folder in ["submissions/20261105", "submissions/20261105_round2"]:
        manifest = read(ROOT/folder/"manifest.json")
        for name, meta in manifest["files"].items():
            if digest(ROOT/folder/name) != meta["sha256"]:
                raise RuntimeError("旧Task1/Task2候选发生变化。")
    for folder, name in [("20261105_task2_round3", "task2_nonlinear_residual.csv"),
                         ("20261105_task2_round4", "task2_robust_refinement.csv")]:
        previous = read(ROOT/"submissions"/folder/"manifest.json")
        if digest(ROOT/"submissions"/folder/name) != previous["submission_sha256"]:
            raise RuntimeError("0.60134/0.59347旧提交发生变化。")
    if digest(ROOT/"task2/outputs/final/submission.csv") != "044c22d6c533e05de661b396b1faf0fb9842c7feb781a806a14346bfd6a035b3":
        raise RuntimeError("最初0.78159提交发生变化。")
    with (SOURCE/"submission.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT/"comp-5434-20261-project-task-2/sampleSubmission.csv").open(encoding="utf-8", newline="") as stream:
        sample = list(csv.DictReader(stream))
    if len(rows) != 10000 or list(rows[0]) != ["id", "rating"] or [r["id"] for r in rows] != [r["id"] for r in sample]:
        raise RuntimeError("提交列名、记录数或id顺序不合法。")
    if not all(math.isfinite(float(r["rating"])) and 1 <= float(r["rating"]) <= 5 for r in rows):
        raise RuntimeError("评分不是有限的1–5连续值。")
    # 测试数取实际运行结果，不将预期数量或单个模块结果冒充完整回归结果。
    suite = unittest.defaultTestLoader.discover(str(ROOT/"task2/tests"))
    with (SOURCE/"regression_tests.log").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Task2完整回归测试失败，详情见regression_tests.log。")
    folder = ROOT/"submissions/20261105_task2_round5"
    folder.mkdir(exist_ok=False)
    target = folder/"task2_semantic_noise_fusion.csv"
    shutil.copy2(SOURCE/"submission.csv", target)
    chosen = read(SOURCE/"frozen_selection.json")
    confirmation = read(SOURCE/"confirmation.json")
    uncertainty = read(SOURCE/"paired_uncertainty.json")
    timing = read(SOURCE/"reproduced.timing.json")
    if timing["semantic_checkpoint_reused"]:
        raise RuntimeError("完整独立复现必须使用新的语义工作目录，不能复用首次预测断点。")
    semantic_timing = read(SOURCE/"semantic_development.timing.json")
    fit_timing = read(SOURCE/"final_timing.json")
    channel_fit = read(SOURCE/"final_channel_fit.json")
    old, new = confirmation["inherited"], confirmation["candidate"]
    old_risk = old["noise_model_risk"]["latent_rmse_estimate"]
    new_risk = new["noise_model_risk"]["latent_rmse_estimate"]
    old_warm = old["all_identities_known_risk"]["latent_rmse_estimate"]
    new_warm = new["all_identities_known_risk"]["latent_rmse_estimate"]
    proof = {"current_known_kaggle_best_user_report": .59347, "new_kaggle_score": None,
        "user_reported_leader_score_unverified": .45700, "records": len(rows),
        "primary_selection_metric": "all_identities_known_risk",
        "frozen_protocol_sha256": digest(SOURCE/"frozen_protocol.json"),
        "id_order_verified": True, "old_submission_hashes_unchanged": True,
        "independent_prediction_byte_identical": True, "tests_passed": result.testsRun,
        "uses_external_pretrained_model": True, "competition_external_model_rule": "not_yet_verified",
        "external_training_overlap_with_competition": "unknown", "selection": chosen,
        "submission_sha256": digest(target), "model_location": "task2/outputs/round5_20261105/model",
        "model_files": {p.relative_to(SOURCE/"model").as_posix(): digest(p) for p in (SOURCE/"model").rglob("*") if p.is_file()},
        "source_code_sha256": {name: digest(ROOT/"task2"/name) for name in
            ["semantic_prepare.py", "semantic_infer.py", "semantic_fusion.py", "noise_softmax.py",
             "semantic_train.py", "semantic_predict.py", "semantic_uncertainty.py"]}}
    (folder/"manifest.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder/"外部模型说明.txt").write_text(
        "本候选使用NLP Town五星BERT的固定Xenova ONNX量化导出作为特征。\n"
        "比赛是否允许外部预训练模型尚未核对。请先核对比赛规则，再决定是否上传本候选。\n"
        "模型外部训练语料与竞赛评论重叠未知；本实现没有下载原始干净评分或查找测试答案。\n"
        "课程实现包含自主Dask词回归、自主噪声通道损失/梯度和自主提升树，详见项目文档。\n", encoding="utf-8")
    lines = ["# Task2 第五轮优化结果（2026-11-05）", "", "## 结论与边界", "",
        "当前已知线上最佳为用户报告的0.59347；本轮尚无新Kaggle分数，用户报告第一名0.45700未独立核验。",
        "本轮引入冻结BERT语义与显式标签噪声通道，旧词模型/历史统计仍继承完整Dask训练流程。",
        "**本候选含外部预训练模型；用户尚未核对比赛规则，上传前需核对是否允许。**",
        "外部模型训练语料与竞赛数据的重叠未知，不能把其验证结果声称为严格的外部语料隔离效果。", "",
        "## 架构与选择", "",
        "冻结模型来源：[NLP Town五星BERT](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)，",
        "采用[Xenova ONNX导出](https://huggingface.co/Xenova/bert-base-multilingual-uncased-sentiment)，固定版本和文件哈希见包内provenance.json。",
        "本机没有微调BERT，也没有把全部768万条评论做BERT前向；57,797条固定哈希样本训练/评估本轮融合器。",
        "输入为标题与正文对，保留词序、最长128个token；原39维+旧评分+19维语义，共59维。",
        "自主通道分类器学习 q=softmax(offset+XW)，观测似然 r=(1-epsilon)q+epsilon/5，评分为五星期望。",
        "对照自主线性校准和400轮深度3残差树；仅内部选择架构、轮数、幅度，再冻结确认。",
        "无标签审计显示验证约44.36%用户无历史，测试三种身份全部有历史。增加有历史专家；",
        "查看任何语义效果之前固定以三种身份均有历史子集的风险为选型主指标，同时完整保留全体指标。",
        f"选定架构：{chosen['architecture']}，相对旧模型融合幅度 {chosen['alpha']}。",
        "原内部预选tree400实际采用语义岭校准+残差树；通道与有历史专家作为对照保存，未被原内部选择。",
        "只读消融后另登记了额外审计；该扩展的选择顺序与结论另见Task2第五轮扩展审计文档，不倒写本预选。",
        "外层组号0内部80/20开发，组号1确认；各轮历史复用已披露，不声称是从未见过的独立测试。", "",
        "## 冻结确认", "", "| 指标 | 0.59347方案对应模型 | 第五轮候选 |", "|---|---:|---:|",
        f"| 原评分RMSE | {old['observed_rmse']:.6f} | {new['observed_rmse']:.6f} |",
        f"| 噪声模型风险RMSE估计 | {old_risk:.6f} | {new_risk:.6f} |",
        f"| 全部身份已有历史子组风险（选型指标） | {old_warm:.6f} | {new_warm:.6f} |",
        f"| 无显式星级标题子组风险估计 | {old['non_explicit_title_risk']['latent_rmse_estimate']:.6f} | {new['non_explicit_title_risk']['latent_rmse_estimate']:.6f} |", "",
        f"确认样本 {new['rows']:,} 条；全体风险RMSE估计相对变化 {(new_risk/old_risk-1)*100:+.2f}%。",
        f"其中三种身份均有历史 {new['all_identities_known_rows']:,} 条，子组风险估计相对变化 {(new_warm/old_warm-1)*100:+.2f}%。",
        "该估计依赖约25%均匀替换假设，不能直接当作线上RMSE，更不能预填新Kaggle分数。",
        f"500次按评论组配对重采样：新减旧 {uncertainty['point_difference']:.6f}，条件95%区间 {uncertainty['conditional_95_interval']}。",
        "区间不包含噪声假设错误、历史选型复用、外部语料重叠或线上分布变化。",
        f"有历史子组的条件95%差异区间：{uncertainty['all_identities_known']['conditional_95_interval']}。",
        "完整只读消融见 confirmation_ablations.json，未根据消融的确认结果重新选择候选。", "",
        "## 成本和复现", "",
        "硬件为i5-11300H、约16GB内存、MX450 2GB；最终语义推理选CPU量化后端，4个ORT线程。",
        "原训练继续使用Dask三进程；本轮语义推理和融合器为本地独立阶段，不冒称分布式BERT训练。",
        f"57,797条语义输入推理 {semantic_timing['inference_seconds_this_run']:.3f} 秒。",
        f"最终残差树训练 {fit_timing['tree_seconds']:.3f} 秒；通道优化 {fit_timing['channel_seconds']:.3f} 秒。",
        f"有历史专家使用 {fit_timing['warm_rows']:,} 条，优化 {fit_timing['warm_channel_seconds']:.3f} 秒。",
        f"通道迭代 {channel_fit['iterations']} 次，优化器成功标志 {channel_fit['success']}，消息 {channel_fit['message']}。",
        f"从原始test.csv完整独立推理10,000条 {timing['complete_inference_seconds']:.3f} 秒，包含语义网络及所有统计/词特征，未复用语义断点。",
        "该次完整复现与额外审计推理并行运行，有CPU资源竞争，不能当作独占CPU速度基准。",
        "以上增量成本不包含原始缓存、Dask词模型及历轮统计训练；不是整个项目从零耗时。",
        "量化输出会随批组成改变，固定批量16、输入长度排序与动态裁剪实现；CPU/GPU输出不混用。",
        f"{result.testsRun}项完整回归测试通过，含通道解析梯度有限差分、验证标签隔离、合成噪声恢复和旧PCG等检查。",
        "完整独立推理与首次CSV逐字节一致；全部旧提交哈希不变；10,000条id顺序、列名、有限1–5评分均通过。", "",
        "## 交付", "", "- 提交候选：submissions/20261105_task2_round5/task2_semantic_noise_fusion.csv。",
        "- 自包含模型：task2/outputs/round5_20261105/model/，包含外部量化权重和所有旧模型依赖。",
        "- 首次执行与独立复现命令：task2/README.md第五轮章节。",
        "- 外部模型规则提示：候选目录/外部模型说明.txt。", ""]
    (ROOT/"docs/Task2第五轮优化结果-20261105.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k:v for k,v in proof.items() if k not in ["model_files", "source_code_sha256", "selection"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
