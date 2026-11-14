"""仅整理第七轮当前方案到已授权的Git副本，不自动提交或推送。

保留Task1所有文件和历史资料，原始数据、大型权重及带身份元数据的缓存不上传。
远端有队友新提交或工作区有改动时停止，以免覆盖他人工作。
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
import task2
from task2.audit import ROOT,sha256,write_json
from task2.history_prepare import OUT


def complete_evidence(pub,variant):
    """只合并Task2的本轮记录，保留发布仓库原有Task1资料；更新当前证据入口。"""
    prefix="sparse_" if variant=="sparse" else ""
    selected=OUT/("sparse_frozen.json" if variant=="sparse" else "frozen_selection.json")
    for source,target in [(selected,"frozen_selection.json"),(OUT/(prefix+"confirmation.json"),"confirmation.json"),
                           (OUT/(prefix+"uncertainty.json"),"paired_uncertainty.json")]:
        shutil.copy2(source,pub/"task2/latest"/target)
    write_json(pub/"task2/latest/protocol.json",{"date":"2026-11-14","variant":variant,
        "known_best_kaggle":.53985,"new_kaggle_score":None,"outer_validation_rows":153456,
        "inner_fit":33437,"inner_check":8268,"confirmation":41342,"final_head_fit":83047,
        "historical_validation_reuse":True,"two_routes_attempted":True,"gpu_used":False,
        "bert_weights_updated":False,"word_classifier_fit_rows":7531189 if variant=="sparse" else None})
    timing=json.loads((OUT/f"reproduced_{variant}.timing.json").read_text(encoding="utf-8"))
    timing["shared_encoder_first_independent_pipeline_seconds"]=415.5705203
    write_json(pub/"task2/latest/reproduced.timing.json",timing)
    (pub/"task2/latest/README.md").write_text("# 当前证据：本地第七轮\n\nprotocol、frozen_selection、confirmation、paired_uncertainty、reproduced.timing和verification_manifest为本轮。\n"
        "linear_anchor_confirmation_read_only和test_parent_verification是继承第六轮父模型审计，不能当作新模型结果。\n"
        "Git不含完整权重；须另取自包含模型包。\n",encoding="utf-8")
    original=subprocess.check_output(["git","-C",str(pub),"show","HEAD:docs/Kaggle测评记录.md"]).decode("utf-8")
    local=(ROOT/"docs/Kaggle测评记录.md").read_text(encoding="utf-8")
    marker="## 2026-11-14：本地第七轮待测"
    addition=local[local.index(marker):].strip()
    (pub/"docs/Kaggle测评记录.md").write_text(original.rstrip()+"\n\n"+addition+"\n",encoding="utf-8")
    write_json(pub/"docs/latest_validation.json",{"date":"2026-11-14","task1_unchanged":True,"task2_tests_passed":37,
        "publication_source_tests_passed":37,"independent_raw_prediction_byte_identical":True,
        "old_submission_files_verified":19,"new_kaggle_score":None,"previous_measured_best":.53985})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--variant",choices=["history","sparse"],required=True);args=parser.parse_args()
    pub=ROOT/"tmp/github-publish-20261108"
    if subprocess.check_output(["git","-C",str(pub),"status","--porcelain"]):raise RuntimeError("发布副本有未提交修改。")
    if subprocess.check_output(["git","-C",str(pub),"rev-list","--left-right","--count","HEAD...origin/main"],text=True).strip()!="0\t0":
        raise RuntimeError("发布副本与已获取远端不同步，先合并队友工作。")
    task1_hash=sha256(pub/"submissions/latest/task1.csv")
    proof=json.loads((ROOT/f"submissions/20261114_task2_round7/manifest_{args.variant}.json").read_text(encoding="utf-8"))
    name="task2_posterior_history.csv" if args.variant=="history" else "task2_sparse_noise_fusion.csv"
    source=ROOT/"submissions/20261114_task2_round7"/name
    assert sha256(source)==proof["sha256"] and proof["independent_raw_csv_prediction"]
    relative=["task2/posterior_history.py","task2/history_prepare.py","task2/history_train.py","task2/sparse_noise.py",
        "task2/sparse_fusion.py","task2/local7_predict.py","task2/finetune_protocol.py","task2/tests/test_posterior_history.py",
        "task2/tests/test_sparse_noise.py","task2/tests/test_finetune.py","task2/docs/本地第七轮原理与复现.md",
        "docs/Task2本地第七轮优化记录-20261114.md","docs/Task2本地第七轮优化结果-20261114.md",
        "docs/Kaggle测评记录.md","tools/finish_task2_local7.py","tools/sync_task2_local7.py"]
    for path in relative:
        target=pub/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,target)
    shutil.copy2(source,pub/"submissions/latest/task2.csv")
    method="后验历史与自主残差提升树" if args.variant=="history" else "Dask自主五星噪声词分类与冻结BERT头融合"
    manifest=json.loads((pub/"submissions/latest/manifest.json").read_text(encoding="utf-8"))
    manifest["snapshot_date"]="2026-11-14"
    manifest["tasks"]["task2"]={**proof,"source_local_path":str(source.relative_to(ROOT)).replace("\\","/"),
        "submitted_file":"submissions/latest/task2.csv","method":method,"model_included":False,"tests_passed":37,
        "latest_kaggle_score":None,"previous_measured_best":.53985}
    write_json(pub/"submissions/latest/manifest.json",manifest)
    title=f"# Task2当前入口：本地第七轮（2026-11-14）\n\n当前候选：{method}。最新CSV为submissions/latest/task2.csv。\n"
    title+="此前第六轮线上0.53985，新候选线上未测评。37项测试通过，结果见根目录本地第七轮报告。\n\n"
    title+="完整模型包需维护者提供，放在task2/latest/model；Git不含大型参数、原始数据或训练缓存。\n\n"
    title+="```powershell\npython -m task2.local7_predict --model task2/latest/model --test comp-5434-20261-project-task-2/test.csv --semantic-python .venv-semantic/Scripts/python.exe --work-dir task2_inference_local7 --output task2_prediction_local7.csv\npython -m unittest discover -s task2/tests -v\n```\n\n"
    title+="原理与复现见docs/本地第七轮原理与复现.md。下方为旧阶段说明，当前入口以上文为准。\n\n---\n\n"
    old=(pub/"task2/README.md").read_text(encoding="utf-8")
    (pub/"task2/README.md").write_text(title+old,encoding="utf-8")
    readme=pub/"README.md";text=readme.read_text(encoding="utf-8")
    text=text.replace("快照整理日期：2026-11-08。","快照整理日期：2026-11-14。")
    before="| Task 2 | 冻结BERT的CLS768维与自主噪声MLP任务头，第六轮候选 | [task2.csv](submissions/latest/task2.csv) | 第六轮待反馈；此前第五轮已测RMSE=0.54604 |"
    if text.count(before)!=1:raise RuntimeError("首页Task2内容变化，需人工保留合并。")
    text=text.replace(before,f"| Task 2 | {method}，本地第七轮 | [task2.csv](submissions/latest/task2.csv) | 新版待测；第六轮已测RMSE=0.53985 |")
    text=text.replace("Task 2第六轮的本地噪声风险估计不等于Kaggle成绩。","Task 2本地第七轮的噪声风险估计不等于Kaggle成绩。")
    text=text.replace("[第六轮结果](docs/Task2第六轮优化结果-20261108.md)","[本地第七轮结果](docs/Task2本地第七轮优化结果-20261114.md)")
    readme.write_text(text,encoding="utf-8")
    prefix="sparse_" if args.variant=="sparse" else ""
    for name in [prefix+"confirmation.json",prefix+"uncertainty.json"]:shutil.copy2(OUT/name,pub/"task2/latest"/name)
    shutil.copy2(ROOT/f"submissions/20261114_task2_round7/manifest_{args.variant}.json",pub/"task2/latest/verification_manifest.json")
    complete_evidence(pub,args.variant)
    assert sha256(pub/"submissions/latest/task1.csv")==task1_hash
    print("当前方案副本准备完成；未执行推送。",args.variant,proof["sha256"],flush=True)


if __name__=="__main__":main()
