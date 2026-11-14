"""依据已产生的真实第八轮文件整理报告，不用预期数值替代运行结果。"""
import json
import task2
from task2.audit import ROOT
from task2.round8_fusion import OUT


def main():
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    scores=json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
    training=json.loads((OUT/"tfidf_split/training.json").read_text(encoding="utf-8"))
    proof=json.loads((ROOT/"submissions/20261114_task2_round8/manifest.json").read_text(encoding="utf-8"))
    ci=json.loads((OUT/"uncertainty.json").read_text(encoding="utf-8"))
    internal=json.loads((OUT/"inner_comparison.json").read_text(encoding="utf-8"))
    final_training=json.loads((OUT/"final_training.json").read_text(encoding="utf-8"))
    joint_linear=min(row["metrics"]["noise_model_risk"]["latent_rmse_estimate"] for row in internal["candidates"] if row["family"]=="joint_linear")
    joint_mlp=min(row["metrics"]["noise_model_risk"]["latent_rmse_estimate"] for row in internal["candidates"] if row["family"]=="joint_mlp")
    final_head_line=(f"最终融合头训练{final_training['iterations']}次L-BFGS迭代，成功收敛={final_training['success']}。"
                     if "iterations" in final_training else "最终联合MLP的锚点和轮次记录见final_training.json。")
    development_timing=json.loads((OUT/"english_development.timing.json").read_text(encoding="utf-8"))
    test_timing=json.loads((OUT/"english_test.timing.json").read_text(encoding="utf-8"))
    rows=[]
    names={"baseline":"第七轮基线（线上0.53719）","candidate":"第八轮冻结候选","without_new_features":"同头移除全部新增输入",
           "without_english":"同头移除英语输入","without_lexical":"同头移除TF-IDF输入","without_english_cls":"同头移除英语CLS"}
    for key,name in names.items():
        if key not in scores:continue
        metric=scores[key]
        rows.append(f"| {name} | {metric['noise_model_risk']['latent_rmse_estimate']:.6f} | {metric['non_explicit_title_risk']['latent_rmse_estimate']:.6f} | {metric['observed_rmse']:.6f} |")
    english="采用" if choice["english"] else "未采用"
    lexical=f"采用{choice['lexical']}迭代权重" if choice["lexical"] else "未采用"
    text=f"""# Task2第八轮优化结果（2026-11-14）

推荐文件：submissions/20261114_task2_round8/task2_round8.csv。第八轮线上尚未测评；此前第七轮线上0.53719。
GitHub快照中同一文件为submissions/latest/task2.csv，SHA-256相同。
本轮全部新增训练/推理在本机CPU完成，没有重新启用云端GPU。

## 实际选定配置

TF-IDF词模型：{lexical}；英语DeBERTa：{english}；头类型={choice.get('family','residual')}，L2={choice['ridge']}，融合系数={choice['alpha']}。
联合线性、联合MLP内部最优风险估计分别为{joint_linear:.6f}、{joint_mlp:.6f}，本次没有胜出；没有因模型更复杂而强行采用。
内部仅用8268条选择，训练33437条；冻结后用41342条确认。最终头拟合83047条。
{final_head_line}
这些验证记录有历史复用，候选并非独立实验测试；所有风险估计依赖固定均匀噪声假设。

| 确认模型 | 噪声风险RMSE估计 | 非显式标题子组 | 原观测RMSE |
|---|---:|---:|---:|
{chr(10).join(rows)}

新减旧风险RMSE差={ci['difference']:.6f}；500次评论组级配对bootstrap的条件95%区间={ci['conditional_95_interval']}。
该区间不包含噪声假设错误、历史/本轮选型偏差、外部语料重叠或分布变化，不是新Kaggle分数。
消融只解释已冻结配置，不根据确认结果重新选参数。

## 词模型及计算边界

只用7531189条训练输入估计IDF，原外层153456条全部排除。标题/正文分别L2归一化，原数值列不变。
Dask真实三进程、每进程一线程训练自主五星噪声通道分类器，L2从50调整为5，40次缩放L-BFGS迭代。
本次训练耗时{training['seconds']:.2f}秒、目标计算{training['objective_evaluations']}次；结束信息：{training['message']}。
不能称为已经收敛，也不能把表示、正则化、额外训练的联合收益全部归于IDF单项。
部署词权重仍只在753万行拟合，测试历史统计使用原full表；没有声称词分类器在全部768万行重训。

## 英语模型实现边界

英语权重固定版本06f020e0dbf909570eb886423bd3af256b855546，MIT，原模型卡训练语料描述不完整，重叠未知。
保留原SentencePiece byte fallback。导出FP32经多输入形状核对；几种INT8均未过真实输入精度门槛，全部拒用。
OpenVINO CPU FP32通过数值实现核对，网络没有微调；它只是候选语义输入，实际是否采用见上方配置。
完整83047条开发编码耗时{development_timing['seconds']:.2f}秒、复用{development_timing['rows_resumed']}行；
10000条测试编码耗时{test_timing['seconds']:.2f}秒、复用{test_timing['rows_resumed']}行。
原理和失败量化说明见task2/docs/DeBERTa与量化原理说明.md，不声称同意使用模型就代表学生已理解。

## 交付核验

10000行id/rating，样例ID顺序、唯一性、1–5有限连续评分通过。
SHA-256：{proof['sha256']}。
原始CSV独立特征路径最大评分差{proof['max_abs_difference']}，逐字节一致={proof['independent_byte_identical']}。
{proof['old_files_unchanged']}个旧提交CSV指纹保持不变。完整41项数值回归和共享常量图独立测试通过。
源码含中文注释，模型包在task2/outputs/round8_20261114/model；Git只更新代码、记录和当前推荐CSV，不含大型权重。

独立推理计时及编码缓存复用情况见reproduced.timing.json；不能把复用编码的时间当作重新编码全流程时间。
仅获得代码仓库不能直接预测，需取得自包含模型包；从零训练还依赖原缓存和前轮产物，二者不能混淆。
"""
    (ROOT/"docs/Task2第八轮优化结果-20261114.md").write_text(text,encoding="utf-8")
    print("第八轮报告已依据实际结果生成",flush=True)


if __name__=="__main__":main()
