"""整理第九轮已核验方案到团队Git副本；不自动提交、推送或操作Kaggle。

保留Task1、父方案小型证据和全部本地旧产物。只复制显式清单中的源码/记录，
不把本地其他Task1实验、身份元数据、大缓存或GPU环境混入上传。
"""
import json
import shutil
import subprocess
import task2
from task2.audit import ROOT,sha256,write_json
from task2.partial9_protocol import OUT,FOLDER,tag,run_name


def git(pub,*args):return subprocess.check_output(["git","-C",str(pub),*args],text=True,encoding="utf-8").strip()


def main():
    pub=ROOT/"tmp/github-publish-20261108"
    if git(pub,"status","--porcelain"):raise RuntimeError("发布副本有改动，先核对。")
    if git(pub,"rev-list","--left-right","--count","HEAD...origin/main")!="0\t0":raise RuntimeError("先合并远端队友工作。")
    proof=json.loads((ROOT/"submissions/20261114_task2_round9/manifest.json").read_text(encoding="utf-8"))
    source=ROOT/"submissions/20261114_task2_round9/task2_round9.csv"
    if not proof["independent_raw_feature_path"] or sha256(source)!=proof["sha256"]:raise ValueError("提交证据不匹配。")
    task1=sha256(pub/"submissions/latest/task1.csv")
    old_record=(pub/"docs/Kaggle测评记录.md").read_text(encoding="utf-8")
    paths=["task2/partial9.py","task2/partial9_protocol.py","task2/partial9_cache.py","task2/partial9_audit.py",
        "task2/partial9_train.py","task2/partial9_fusion.py","task2/partial9_infer.py","task2/partial9_cls.py","task2/partial9_head_control.py","task2/round9_predict.py",
        "task2/round8_predict.py","task2/docs/第九轮最后层适配与复现.md",
        "docs/Task2第九轮优化记录-20261114.md","docs/Task2第九轮优化结果-20261114.md",
        "tools/download_round9_cuda.py","tools/verify_partial9_math.py","tools/smoke_partial9_gpu.py",
        "tools/finish_task2_round9.py","tools/report_task2_round9.py","tools/sync_task2_round9.py","tools/check_partial9_deployment.py",
        "tools/check_partial9_prediction_batch.py","tools/complete_task2_round9.py","tools/check_partial9_cpu_threads.py","tools/enable_partial9_cpu_threads.py","tools/recover_task2_round9_threads.py",
        "task2/title10.py","task2/title10_encode.py","task2/round10_predict.py","task2/tests/test_title10.py","task2/english_infer.py","tools/cache_title10_test_gpu.py","tools/verify_title10_cache.py",
        "task2/docs/第十轮标题语义与复现.md","task2/docs/阶段记录.md","docs/Task2第十轮优化记录-20261114.md",
        "tools/complete_task2_title10.py","tools/report_task2_title10.py","tools/sync_task2_title10.py","tools/finalize_task2_round10_docs.py","tools/record_task2_publication10.py"]
    for relative in paths:
        target=pub/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
    latest=pub/"task2/latest";history=latest/"parent_round8";history.mkdir(exist_ok=False)
    for previous in list(latest.iterdir()):
        if previous==history:continue
        target=history/previous.name
        # Windows目录移动前同时核对绝对源/目标都在明确的latest范围，保护父目录和Task1。
        if latest.resolve() not in previous.resolve().parents or latest.resolve() not in target.resolve().parents:
            raise ValueError("父证据归档路径越界。")
        shutil.move(str(previous),str(target))
    aliases={"selection_protocol.json":"selection_protocol.json","selection_protocol_initial24.json":"initial_protocol.json",
        "frozen_selection.json":"frozen_selection.json","inner_comparison.json":"inner_comparison.json",
        "confirmation.json":"confirmation.json","uncertainty.json":"paired_uncertainty.json","source.json":"source.json",
        "split_precision_audit.json":"split_precision_audit.json","local_gpu_audit.json":"local_gpu_audit.json",
        "deterministic_gpu_smoke.json":"deterministic_gpu_smoke.json","baseline_internal_error_audit.json":"baseline_internal_error_audit.json",
        "reproduced.timing.json":"reproduced.timing.json","final_fusion_training.json":"final_fusion_training.json",
        "model_reference.json":"model_reference.json"}
    aliases["adapted_deployment_audit.json"]="adapted_deployment_audit.json"
    aliases["gpu_prediction_batch_audit.json"]="gpu_prediction_batch_audit.json"
    aliases["gpu_prediction_batch_mixed_first.json"]="gpu_prediction_batch_mixed_first.json"
    aliases["completion_pipeline.json"]="completion_pipeline.json"
    aliases["cpu_threads_audit.json"]="cpu_threads_audit.json"
    aliases["cpu_threads_full_test_audit.json"]="cpu_threads_full_test_audit.json"
    aliases["test_prefix_fresh_timing.json"]="test_prefix_fresh_timing.json"
    aliases["completion_first_attempt.json"]="completion_first_attempt.json"
    for name in ["selection_protocol_before_teacher48.json","selection_protocol_before_cls72.json",
                 "selection_protocol_before_residual96.json","first_snapshot_readonly_diagnostic.json",
                 "first_cls_head_readonly_probe.json","new_math_tests.json","regression_tests.json"]:
        aliases[name]=name
    for original,target in aliases.items():shutil.copy2(OUT/original,latest/target)
    for name in ["split_precision_audit_256.json","local_gpu_audit_256.json"]:
        if (OUT/name).exists():shutil.copy2(OUT/name,latest/name)
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    shutil.copy2(FOLDER/run_name("final",choice)/"training.json",latest/"neural_training.json")
    shutil.copy2(FOLDER/run_name("confirm",choice)/"training.json",latest/"confirm_neural_training.json")
    shutil.copy2(FOLDER/run_name("confirm",choice,True)/"training.json",latest/"confirm_classifier_only_training.json")
    shutil.copy2(FOLDER/"model/model.json",latest/"model.json")
    shutil.copy2(FOLDER/"model/adapter/runtime.json",latest/"adapter_runtime.json")
    protocol=json.loads((OUT/"selection_protocol.json").read_text(encoding="utf-8"))
    write_json(latest/"inner_neural_training.json",{run_name("inner",item):json.loads((FOLDER/run_name("inner",item)/"training.json").read_text(encoding="utf-8")) for item in protocol["objective_runs"]})
    shutil.copy2(FOLDER/"development/timing.json",latest/"prefix_training_input.timing.json")
    shutil.copy2(FOLDER/"model/file_integrity.json",latest/"model_file_integrity.json")
    shutil.copy2(ROOT/"submissions/20261114_task2_round9/manifest.json",latest/"verification_manifest.json")
    write_json(latest/"protocol.json",{"round":9,"date":"2026-11-14","previous_measured_best":.50646,"new_kaggle_score":None,
        "inner_fit":33437,"inner_check":8268,"confirmation":41342,"final_neural_fit":83047,
        "frozen_encoder_layers":11,"adapted_encoder_layers":1,"training_max_length":128,"inference_max_length":choice["max_length"],
        "historical_validation_reuse":True,"local_gpu_used":True,"cloud_gpu_used":False})
    (latest/"README.md").write_text("# 当前证据：第九轮最后层适配\n\n"
        "当前protocol、frozen_selection、confirmation、paired_uncertainty、verification_manifest和训练/数值审计均为第九轮。\n"
        "第八轮线上0.50646，新分数待反馈；本地风险不等于线上RMSE。父轮记录完整保存在parent_round8。\n"
        "Git不含完整Task2模型，需另取整个model包，按task2/README开头在CPU预测。\n",encoding="utf-8")
    shutil.copy2(source,pub/"submissions/latest/task2.csv")
    manifest=json.loads((pub/"submissions/latest/manifest.json").read_text(encoding="utf-8"));manifest["snapshot_date"]="2026-11-14"
    manifest["tasks"]["task2"]={**proof,"source_local_path":"submissions/20261114_task2_round9/task2_round9.csv",
        "submitted_file":"submissions/latest/task2.csv","method":"英语DeBERTa最后层噪声微调与第八轮融合",
        "choice":choice,"model_included":False,"latest_kaggle_score":None,"previous_measured_best":.50646}
    write_json(pub/"submissions/latest/manifest.json",manifest)
    header="# Task2当前入口：第九轮最后层适配\n\n最新CSV：submissions/latest/task2.csv。第八轮已测0.50646，第九轮待测。\n\n"
    header+="英语模型前11层冻结，第12层和原分类头用课程评分做噪声训练；本机GPU训练，CPU可预测。\n"
    header+="实际配置/确认结果见根目录docs/Task2第九轮优化结果-20261114.md，理解和复现见docs/第九轮最后层适配与复现.md。\n\n"
    header+="Git不含大型权重，需另取完整model文件夹，放task2/latest/model。英语CPU环境还需官方CPU版torch2.6.0。\n\n"
    header+="```powershell\npython -m task2.round9_predict --model task2/latest/model --test comp-5434-20261-project-task-2/test.csv --semantic-python .venv-semantic/Scripts/python.exe --english-python .venv-english/Scripts/python.exe --work-dir task2_inference_round9 --output task2_prediction_round9.csv\npython -m unittest discover -s task2/tests -v\n```\n\n下方为历史阶段说明，当前入口以上文为准。\n\n---\n\n"
    readme=pub/"task2/README.md";readme.write_text(header+readme.read_text(encoding="utf-8"),encoding="utf-8")
    top=pub/"README.md";lines=top.read_text(encoding="utf-8").splitlines();where=[i for i,v in enumerate(lines) if v.startswith("| Task 2 |")]
    if len(where)!=1:raise RuntimeError("首页入口变化，需保留式合并。")
    lines[where[0]]="| Task 2 | 英语DeBERTa最后层噪声微调与第八轮融合，第九轮 | [task2.csv](submissions/latest/task2.csv) | 第九轮待测；第八轮已测RMSE=0.50646 |"
    text="\n".join(lines)+"\n";text=text.replace("Task 2第八轮的噪声风险估计","Task 2第九轮的噪声风险估计")
    text=text.replace("[第八轮结果](docs/Task2第八轮优化结果-20261114.md)","[第九轮结果](docs/Task2第九轮优化结果-20261114.md)")
    text=text.replace("本轮约1.2GB的完整模型包","本轮约2GB的完整模型包");top.write_text(text,encoding="utf-8")
    local=(ROOT/"docs/Kaggle测评记录.md").read_text(encoding="utf-8");marker="## 2026-11-14：第八轮线上反馈与第九轮"
    (pub/"docs/Kaggle测评记录.md").write_text(old_record.rstrip()+"\n\n"+local[local.index(marker):].strip()+"\n",encoding="utf-8")
    upload=pub/"docs/最新上传说明.md";upload.write_text("# 当前更新：Task2第九轮\n\nTask2代码、中文阶段记录、审计和当前CSV已更新。"
        "第八轮已测0.50646，新版线上待测。Task1继续沿用仓库原版。\n完整约2GB参数包需另取，Git不含大模型、数据或训练缓存。\n\n以下为历史说明。\n\n---\n\n"+upload.read_text(encoding="utf-8"),encoding="utf-8")
    write_json(pub/"docs/latest_validation.json",{"round":9,"date":"2026-11-14","task1_unchanged":True,
        "numerical_tests":41,"new_math_tests":7,"deterministic_gpu_smoke_passed":True,"publication_tests_passed":None,
        "independent_raw_feature_prediction_byte_identical":proof["independent_byte_identical"],
        "old_submission_files_verified":proof["old_files_unchanged"],"new_kaggle_score":None,"previous_measured_best":.50646})
    if sha256(pub/"submissions/latest/task1.csv")!=task1:raise RuntimeError("Task1指纹变化。")
    print("第九轮Git副本整理完成，待在实际副本中验证再提交。",proof["sha256"],flush=True)


if __name__=="__main__":main()
