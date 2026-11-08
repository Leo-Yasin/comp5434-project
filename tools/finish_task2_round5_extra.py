"""第五轮扩展实验报告与可选通道候选交付，不覆盖预选结果，不执行线上提交。"""
import csv
import json
import math
import shutil
from pathlib import Path

from task2.audit import ROOT, sha256, write_json


def main():
    source = ROOT/"task2/outputs/round5_20261105"
    folder = ROOT/"submissions/20261105_task2_round5"
    audit = json.loads((source/"extra_audit.json").read_text(encoding="utf-8"))
    released = audit["release_channel_as_additional_candidate"]
    a, b, baseline = audit["tree400"], audit["channel"], audit["baseline"]
    write_json(folder/"extra_audit_manifest.json", {"channel_released": released,
        "frozen_extra_protocol_sha256": sha256(source/"extra_frozen_protocol.json"),
        "extra_audit_sha256": sha256(source/"extra_audit.json"),
        "semantic_extra_predictions_sha256": sha256(source/"semantic_extra.npz"),
        "source_code_sha256": {name: sha256(ROOT/"task2"/name) for name in
            ["semantic_extra_audit.py", "semantic_channel_predict.py"]}})
    lines = ["# Task2第五轮扩展审计", "", "## 目的与顺序", "",
        "原预选tree400通过确认，照常保留，不能把后续试验倒写成原预选结果。",
        "首次确认的只读消融提示全体通道模型可能更好，因此先固定两候选与额外哈希区间，",
        "再对另一批9,447条评论推理。该批不参与本轮融合训练，未改参数或融合权重。",
        "它仍属于历史外层验证，不能称为完全独立的新测试。", "",
        "## 额外冻结比较", "", "| 指标 | 旧0.59347对应模型 | 预选语义残差树 | 语义通道模型 |",
        "|---|---:|---:|---:|",
        f"| 原观测评分RMSE | {baseline['observed_rmse']:.6f} | {a['observed_rmse']:.6f} | {b['observed_rmse']:.6f} |",
        f"| 全体风险RMSE估计 | {baseline['noise_model_risk']['latent_rmse_estimate']:.6f} | {a['noise_model_risk']['latent_rmse_estimate']:.6f} | {b['noise_model_risk']['latent_rmse_estimate']:.6f} |",
        f"| 全部身份有历史子组风险 | {baseline['all_identities_known_risk']['latent_rmse_estimate']:.6f} | {a['all_identities_known_risk']['latent_rmse_estimate']:.6f} | {b['all_identities_known_risk']['latent_rmse_estimate']:.6f} |", "",
        f"有历史子组 {b['all_identities_known_rows']:,} 条；通道减树的风险RMSE差 {audit['channel_minus_tree_known']['difference']:.6f}，",
        f"500次组重采样条件95%区间 {audit['channel_minus_tree_known']['conditional_95_interval']}。",
        "区间依赖固定均匀替换假设；不包含历史选择复用、外部训练重叠、假设误差或线上分布变化。",
        "上述所有数值均为本地假设风险/观测验证，不能称为新的Kaggle实测。", ""]
    if released:
        first = source/"channel_submission.csv"
        reproduced = source/"channel_reproduced.csv"
        if first.read_bytes() != reproduced.read_bytes():
            raise RuntimeError("通道候选独立预测不一致。")
        with first.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with (ROOT/"comp-5434-20261-project-task-2/sampleSubmission.csv").open(encoding="utf-8", newline="") as stream:
            sample = list(csv.DictReader(stream))
        if len(rows) != 10000 or list(rows[0]) != ["id", "rating"] or [r["id"] for r in rows] != [r["id"] for r in sample]:
            raise RuntimeError("通道提交列名/记录数/id顺序不合法。")
        if not all(math.isfinite(float(r["rating"])) and 1 <= float(r["rating"]) <= 5 for r in rows):
            raise RuntimeError("通道提交评分非法。")
        target = folder/"task2_semantic_channel.csv"
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(first, target)
        write_json(folder/"channel_manifest.json", {"current_known_best_kaggle_user_report": .59347,
            "new_kaggle_score": None, "extra_stage_chosen_after_original_confirmation": True,
            "extra_audit_parameters_registered_before_evaluation": True,
            "original_frozen_tree_candidate_preserved": True,
            "independent_prediction_byte_identical": True, "records": len(rows),
            "uses_external_pretrained_model": True, "competition_external_model_rule": "not_yet_verified",
            "submission_sha256": sha256(target),
            "shared_model_package": "task2/outputs/round5_20261105/model",
            "candidate_description": "channel_candidate.json; original model.json still selects tree400",
            "frozen_extra_protocol_sha256": sha256(source/"extra_frozen_protocol.json"),
            "extra_audit_sha256": sha256(source/"extra_audit.json"),
            "source_code_sha256": {name: sha256(ROOT/"task2"/name) for name in
                ["semantic_extra_audit.py", "semantic_channel_predict.py"]}})
        lines += ["## 扩展结论与交付", "",
            "通道模型达到预登记发布条件：有历史子组风险更低，且配对差异区间上界小于0。",
            "因此另交付task2_semantic_channel.csv。原tree400文件完整保留，不冒充原预选就是通道。",
            "如果比赛允许外部预训练模型，优先测通道候选；语义残差树留作第二候选。",
            "通道在原28,879条确认和这9,447条额外区间均有对照依据，仍不能保证线上改善。",
            "原始测试CSV+第一次语义输出和独立重跑语义输出分别生成通道文件，逐字节一致；",
            "原始字段文本指纹、id与外部权重指纹均检查通过，包内包含全部参数和审计说明。", ""]
    else:
        lines += ["## 扩展结论", "",
            "通道没有满足预登记的额外发布条件，保持原预选tree400，不增加需上传的通道文件。", ""]
    lines += ["## 限制", "",
        "外部模型规则尚未核对，预训练语料与竞赛数据重叠未知；上传前需要核对比赛规则。",
        "当前已知线上最佳仍为用户报告的0.59347，本轮新候选尚未上传。", ""]
    (ROOT/"docs/Task2第五轮扩展审计-20261105.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"channel_released": released, "extra_audit": audit["channel_minus_tree_known"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
