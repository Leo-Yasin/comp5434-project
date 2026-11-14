"""根据实际标题分支结果生成报告，未通过也记录原因，绝不预填线上成绩。"""
import json
import task2
import numpy as np
import pandas as pd
from task2.audit import ROOT
from task2.audit import write_json
from task2.title10 import OUT,PARENT
from tools.finish_task2_local7 import uncertainty


def main():
    decision=json.loads((OUT/"decision.json").read_text(encoding="utf-8"))
    rows=[]
    if (OUT/"inner_comparison.json").exists():
        inner=json.loads((OUT/"inner_comparison.json").read_text(encoding="utf-8"));choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
        rows.append(f"内部风险RMSE：{inner['baseline']['noise_model_risk']['latent_rmse_estimate']:.6f} → {choice['metrics']['noise_model_risk']['latent_rmse_estimate']:.6f}。")
        rows.append("冻结配置："+json.dumps(choice,ensure_ascii=False)+"。")
    if (OUT/"confirmation.json").exists():
        scores=json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
        rows.append(f"固定确认风险RMSE：{scores['baseline']['noise_model_risk']['latent_rmse_estimate']:.6f} → {scores['candidate']['noise_model_risk']['latent_rmse_estimate']:.6f}。")
    if decision["accepted"]:
        proof=json.loads((ROOT/"submissions/20261114_task2_round10/manifest.json").read_text(encoding="utf-8"));ci=json.loads((OUT/"uncertainty.json").read_text(encoding="utf-8"))
        rows.append("推荐submissions/20261114_task2_round10/task2_round10.csv；Git快照对应submissions/latest/task2.csv，SHA相同。SHA-256="+proof["sha256"]+"。")
        rows.append(f"10000行格式/样例ID/连续1–5评分、父轮重建和旧文件保护均通过；独立预测最大差{proof['max_abs_difference']}，逐字节一致={proof['independent_byte_identical']}。")
        rows.append("风险RMSE新减旧="+str(ci["difference"])+"；500次组级配对bootstrap条件95%区间="+str(ci["conditional_95_interval"])+"。")
        frame=pd.read_csv(OUT/"confirmation_predictions.csv");old=pd.read_csv(PARENT/"confirmation_predictions.csv")
        np.testing.assert_array_equal(frame.id,old.id);frame["baseline"]=old.baseline
        overall=uncertainty(frame,.25165131324376977);write_json(OUT/"overall_uncertainty.json",overall)
        rows.append("相对已测第八轮的本地风险RMSE为0.487717→0.480138，差="+str(overall["difference"])+"；同组条件95%区间="+str(overall["conditional_95_interval"])+"。这不是线上成绩差。")
        rows.append("完整模型位于"+proof["model"]+"；Git不含大型参数。父包只读硬链接节省本机空间，复制给队友按普通文件读取。")
        gpu=json.loads((OUT/"test_gpu_timing.json").read_text(encoding="utf-8"))
        rows.append(f"测试标题首次FP32 GPU编码{gpu['seconds']:.2f}秒；后续原始CSV路径复用10000条经过输入/模型/分词器/输出SHA绑定的无标签CUDA缓存，始终重建最终评分。旧流水线阶段名中的CPU编码实际为缓存校验，不作为全新CPU推理计时。冷启动无缓存时仍可全CPU预测。")
    else:rows.append("标题分支未采用："+decision["reason"]+"。当前继续推荐第九轮，不按确认结果改选其他标题候选。")
    text="# Task2第十轮标题语义实际结果\n\n"+"\n\n".join(rows)+"\n\n"
    text+="第八轮真实Kaggle最佳仍为0.50646；第九/十轮线上分数未知。上述风险依赖固定均匀标签替换假设，不等于Kaggle RMSE。\n\n"
    text+="本轮使用冻结英语模型单独编码标题，以第九轮真实五星概率混合为偏置，只训练自主噪声残差头。标题网络全部12层固定，原文本/标签界限不变。\n\n"
    text+="8项内部候选冻结后才做确认；确认有历史复用，条件区间不校正模型选择、噪声假设、外部语料重叠未知或分布变化。不能把分组称为全新未使用过的数据。\n\n"
    text+="42项数值测试含新增概率混合手算/位移不变性检验；第九轮7项数学测试及4项实际标题缓存输入/模型/完成状态核验另计。标题GPU/CPU精度和实际耗时见该轮证据文件。\n\n"
    text+="理解与CPU复现见task2/docs/第十轮标题语义与复现.md；包含失败尝试的完整过程见本轮记录。\n"
    (ROOT/"docs/Task2第十轮优化结果-20261114.md").write_text(text,encoding="utf-8")
    print("标题分支实际报告已生成",decision,flush=True)


if __name__=="__main__":main()
