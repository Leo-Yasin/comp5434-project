"""先整理第九轮，再按实际标题决策更新最新版；Task1始终保留原Git状态。"""
import json
import shutil
import task2
from task2.audit import ROOT,sha256,write_json
from task2.title10 import OUT,FOLDER
from tools.sync_task2_round9 import main as sync_parent


def main():
    decision=json.loads((OUT/"decision.json").read_text(encoding="utf-8"))
    sync_parent();pub=ROOT/"tmp/github-publish-20261108"
    task1=sha256(pub/"submissions/latest/task1.csv")
    shutil.copy2(ROOT/"docs/Task2第十轮优化结果-20261114.md",pub/"docs/Task2第十轮优化结果-20261114.md")
    if not decision["accepted"]:
        evidence=pub/"task2/latest/title_experiment";evidence.mkdir(exist_ok=False)
        for name in ["protocol.json","decision.json","completion.json","gpu_fidelity.json","encoding_timing.json","inner_comparison.json","frozen_selection.json","confirmation.json"]:
            if (OUT/name).exists():shutil.copy2(OUT/name,evidence/name)
        with (pub/"README.md").open("a",encoding="utf-8") as stream:stream.write("\n第十轮标题语义独立实验未采用，原因与对照见[实际结果](docs/Task2第十轮优化结果-20261114.md)；当前CSV继续为第九轮。\n")
        print("标题分支未采用，已上传实验记录，当前CSV仍为第九轮。",flush=True);return
    proof=json.loads((ROOT/"submissions/20261114_task2_round10/manifest.json").read_text(encoding="utf-8"))
    latest=pub/"task2/latest";history=latest/"parent_round9";history.mkdir(exist_ok=False)
    for previous in list(latest.iterdir()):
        if previous==history:continue
        target=history/previous.name
        if latest.resolve() not in previous.resolve().parents or latest.resolve() not in target.resolve().parents:raise ValueError("归档绝对路径越界。")
        shutil.move(str(previous),str(target))
    for name in ["protocol.json","decision.json","completion.json","gpu_fidelity.json","encoding_timing.json","input_binding.json",
                 "inner_comparison.json","frozen_selection.json","confirmation.json","confirm_training.json","final_training.json",
                 "uncertainty.json","overall_uncertainty.json","model_reference.json","reproduced.timing.json","final_title_deployment_audit.json","test_gpu_timing.json","cache_protocol_tests.json"]:shutil.copy2(OUT/name,latest/name)
    shutil.copy2(OUT/"completion_first_attempt.json",latest/"completion_first_attempt.json")
    shutil.copy2(FOLDER/"model/file_integrity.json",latest/"model_file_integrity.json");shutil.copy2(FOLDER/"model/model.json",latest/"model.json")
    shutil.copy2(OUT/"uncertainty.json",latest/"paired_uncertainty.json")
    shutil.copy2(ROOT/"submissions/20261114_task2_round10/manifest.json",latest/"verification_manifest.json")
    source=ROOT/"submissions/20261114_task2_round10/task2_round10.csv"
    if sha256(source)!=proof["sha256"]:raise ValueError("标题提交指纹不符。")
    shutil.copy2(source,pub/"submissions/latest/task2.csv")
    manifest=json.loads((pub/"submissions/latest/manifest.json").read_text(encoding="utf-8"))
    manifest["tasks"]["task2"]={**proof,"submitted_file":"submissions/latest/task2.csv","source_local_path":"submissions/20261114_task2_round10/task2_round10.csv",
        "method":"冻结标题语义与第九轮正文/历史模型的自主噪声残差","model_included":False,"latest_kaggle_score":None}
    write_json(pub/"submissions/latest/manifest.json",manifest)
    (latest/"README.md").write_text("# 当前证据：第十轮标题语义\n\n当前8候选冻结、确认、编码精度、耗时及核验文件均为第十轮。第九轮完整证据保存在parent_round9。\n第八轮已测0.50646；第九/十轮待测。Git不含完整参数，另取模型后按第十轮指南预测。\n",encoding="utf-8")
    path=pub/"task2/README.md";path.write_text("# 当前入口：第十轮冻结标题语义\n\n最新CSV：submissions/latest/task2.csv。第八轮已测0.50646，第十轮待测。\n\n"
        "实际结果见根目录docs/Task2第十轮优化结果-20261114.md；原理和命令见docs/第十轮标题语义与复现.md。完整参数另取，放task2/latest/model。\n\n"
        "```powershell\npython -m task2.round10_predict --model task2/latest/model --test comp-5434-20261-project-task-2/test.csv --semantic-python .venv-semantic/Scripts/python.exe --english-python .venv-english/Scripts/python.exe --work-dir task2_inference_round10 --output task2_prediction_round10.csv\n```\n\n下方为父轮/历史说明。\n\n---\n\n"+path.read_text(encoding="utf-8"),encoding="utf-8")
    top=pub/"README.md";lines=top.read_text(encoding="utf-8").splitlines()
    for i,line in enumerate(lines):
        if line.startswith("| Task 2 |"):
            lines[i]="| Task 2 | 冻结标题语义与正文/历史噪声残差融合，第十轮 | [task2.csv](submissions/latest/task2.csv) | 第十轮待测；第八轮已测RMSE=0.50646 |"
    top.write_text("\n".join(lines).replace("[第九轮结果](docs/Task2第九轮优化结果-20261114.md)","[第十轮结果](docs/Task2第十轮优化结果-20261114.md)").replace("Task 2第九轮的噪声风险估计","Task 2第十轮的噪声风险估计")+"\n",encoding="utf-8")
    upload=pub/"docs/最新上传说明.md"
    upload.write_text("# 当前更新：Task2第十轮\n\n已更新冻结标题语义、中文记录、8候选确认与独立预测证据；第九轮作为父轮完整保留。第八轮已测0.50646，新版待测。Task1原样保留。\n\n以下为历史说明。\n\n---\n\n"+upload.read_text(encoding="utf-8"),encoding="utf-8")
    validation=json.loads((pub/"docs/latest_validation.json").read_text(encoding="utf-8"));validation.update(round=10,numerical_tests=42,publication_tests_passed=None,
        independent_raw_feature_prediction_byte_identical=proof["independent_byte_identical"],old_submission_files_verified=proof["old_files_unchanged"]);write_json(pub/"docs/latest_validation.json",validation)
    if sha256(pub/"submissions/latest/task1.csv")!=task1:raise ValueError("Task1指纹变化。")
    print("第十轮Git副本整理完成，等待实际副本测试与独立预测。",proof["sha256"],flush=True)


if __name__=="__main__":main()
