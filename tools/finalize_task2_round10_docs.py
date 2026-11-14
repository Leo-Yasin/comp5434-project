"""只在两轮交付证据齐全后更新中文索引，保护Task1说明及所有旧记录。"""
import json
from pathlib import Path
import shutil
import task2
from task2.audit import ROOT,write_json,sha256
from task2.title10 import OUT,PARENT


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def main():
    p9=read(ROOT/"submissions/20261114_task2_round9/manifest.json")
    decision=read(OUT/"decision.json");accepted=decision["accepted"]
    p10=read(ROOT/"submissions/20261114_task2_round10/manifest.json") if accepted else None
    if not p9["independent_raw_feature_path"] or (accepted and not p10["independent_raw_feature_path"]):raise ValueError("尚未完成独立交付核验。")
    if accepted:
        model=Path(p10["model"]);integrity=model/"file_integrity.json";before=read(integrity)
        guide=ROOT/"task2/docs/第十轮标题语义与复现.md"
        if (model/"UNDERSTANDING.md").read_bytes()!=guide.read_bytes():
            write_json(OUT/"model_integrity_before_document_alignment.json",before)
            shutil.copy2(guide,model/"UNDERSTANDING.md")
            after={str(p.relative_to(model)).replace("\\","/"):{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(model.rglob("*")) if p.is_file() and p!=integrity}
            if any(after[name]!=value for name,value in before.items() if name!="UNDERSTANDING.md"):raise ValueError("说明对齐改变了其他参数文件。")
            write_json(integrity,after)
            ref=read(OUT/"model_reference.json");ref["logical_bytes"]=sum(p.stat().st_size for p in model.rglob("*") if p.is_file());write_json(OUT/"model_reference.json",ref)
    # 当前运行的旧阶段名写作CPU新编码，实际采用新显式协议；保留原名称，校正含义。
    completion=read(OUT/"completion.json")
    if accepted:
        for step in completion["steps"]:
            if step["name"]=="测试标题全新CPU编码":
                step.update(original_label=step["name"],name="测试标题受核验冻结缓存",actual_backend="verified_torch_cuda_cache",new_network_rows=0,fresh_gpu_timing="test_gpu_timing.json")
        write_json(OUT/"completion.json",completion)
    log=ROOT/"docs/Kaggle测评记录.md";text=log.read_text(encoding="utf-8")
    marker="第九轮实际交付（冻结配置）"
    if marker in text:raise RuntimeError("文档已经更新，不重复追加。")
    text+="\n\n"+marker+"：submissions/20261114_task2_round9/task2_round9.csv，SHA="+p9["sha256"]+"。\n"
    text+="固定确认风险RMSE为0.487717→0.484804，条件95%区间跨零；控制不支持把收益全部归因于编码层微调。第九轮线上分数未知。\n"
    if accepted:
        text+="\n第十轮独立标题8候选已通过：submissions/20261114_task2_round10/task2_round10.csv，SHA="+p10["sha256"]+"。\n"
        text+="固定确认风险RMSE为0.484804→0.480138，采用低维标题头L2=1000、融合1.0。该版本为新的待测候选，真实线上已知最佳仍为第八轮0.50646。\n"
        text+="测试标题首次由GPU生成冻结表示，CPU/原始CSV路径校验严格来源后复用；最终父评分和新评分独立重建。实际耗时与缓存协议测试单独保存。\n"
    else:text+="\n第十轮标题实验未采用，原因见对应结果报告；继续推荐第九轮，线上分数未知。\n"
    log.write_text(text,encoding="utf-8")
    top=ROOT/"README.md";text=top.read_text(encoding="utf-8")
    text=text.replace("[推荐CSV](submissions/20261114_task2_round8/task2_round8.csv)","[已测CSV](submissions/20261114_task2_round8/task2_round8.csv)")
    anchor="当前已知线上最佳为第八轮0.50646"
    where=text.index(anchor)
    row9="| 第九轮 | 最后编码层残差适配与256上下文 | 待测 | [记录](docs/Task2第九轮优化记录-20261114.md) / [结果](docs/Task2第九轮优化结果-20261114.md) | [CSV](submissions/20261114_task2_round9/task2_round9.csv) |\n"
    row10="| 第十轮 | 冻结标题语义与父轮噪声残差 | 待测 | [记录](docs/Task2第十轮优化记录-20261114.md) / [结果](docs/Task2第十轮优化结果-20261114.md) | [推荐CSV](submissions/20261114_task2_round10/task2_round10.csv) |\n" if accepted else ""
    # 现有表格结束后的空行之前插入新行，其他Task1区块一字不改。
    start=text[:where].rstrip();text=start+"\n"+row9+row10+"\n"+text[where:];top.write_text(text,encoding="utf-8")
    round_=10 if accepted else 9;predictor="round10_predict" if accepted else "round9_predict"
    readme=ROOT/"task2/README.md"
    header=f"# Task2当前入口：第{round_}轮\n\n最新待测CSV：submissions/20261114_task2_round{round_}/task2_round{round_}.csv。\n"
    header+="第八轮已测0.50646，新版本线上未知。本地噪声风险估计与Kaggle分数分开记录。完整模型包位于C盘，位置见该轮model_reference.json，Git不含大参数。\n\n"
    header+=f"CPU原始CSV预测入口：python -m task2.{predictor}；完整参数和环境见对应轮次理解与复现文档。\n\n下方为历史说明。\n\n---\n\n"
    readme.write_text(header+readme.read_text(encoding="utf-8"),encoding="utf-8")
    phase=ROOT/"task2/docs/阶段记录.md"
    with phase.open("a",encoding="utf-8") as stream:
        stream.write("\n## 2026-11-14：第九轮及独立标题分支交付\n\n第九轮完成4组实际最后编码层适配、128候选内部比较及冻结确认，风险RMSE 0.487717→0.484804，消融和不确定性如实记录。\n")
        stream.write("第十轮标题8候选"+("通过，固定确认0.484804→0.480138，推荐新CSV。" if accepted else "未采用，继续第九轮。")+"完整格式、原始CSV独立预测、父轮重建和旧提交指纹保护通过；真实线上最佳仍为0.50646。\n")
    if accepted:
        with (ROOT/"docs/Task2第十轮优化记录-20261114.md").open("a",encoding="utf-8") as stream:
            gpu=read(OUT/"test_gpu_timing.json")
            stream.write(f"\n## 测试标题执行来源校正\n\n首次测试标题FP32 GPU前向耗时{gpu['seconds']:.2f}秒、未复用。原始CPU阶段实际执行受核验CUDA冻结缓存协议，复用10000行；已将completion名称校正并保留original_label。来源/输入/网络/分词器/输出SHA及完成状态均校验，4项协议核验通过。最终评分重算；不把缓存校验秒数作为全新CPU编码速度。\n")
    write_json(OUT/"documentation_update.json",{"updated":True,"recommended_round":round_,"task1_sections_preserved":True})
    print("中文索引更新完成，当前推荐轮次",round_,flush=True)


if __name__=="__main__":main()
