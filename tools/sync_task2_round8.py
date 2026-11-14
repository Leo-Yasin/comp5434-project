"""整理已验证的第八轮到团队发布副本；不在本脚本中提交、推送或调用Kaggle。

仅复制明确的Task2源码/小型证据与一个最新CSV。Task1和远端队友改动受保护，
历史父模型证据移动到具名目录，避免把旧指标当成当前指标。运行前需先fetch。
"""
import json
import shutil
import subprocess
from pathlib import Path
import task2
from task2.audit import ROOT,sha256,write_json
from task2.round8_fusion import OUT


def git(pub,*args):
    return subprocess.check_output(["git","-C",str(pub),*args],text=True,encoding="utf-8").strip()


def main():
    pub=ROOT/"tmp/github-publish-20261108"
    if git(pub,"status","--porcelain"):raise RuntimeError("发布副本有修改，请先核对而非覆盖。")
    if git(pub,"rev-list","--left-right","--count","HEAD...origin/main")!="0\t0":
        raise RuntimeError("需先合并最新队友修改；不能覆盖远端。")
    proof=json.loads((ROOT/"submissions/20261114_task2_round8/manifest.json").read_text(encoding="utf-8"))
    source=ROOT/"submissions/20261114_task2_round8/task2_round8.csv"
    if sha256(source)!=proof["sha256"] or not proof["independent_raw_feature_path"]:
        raise ValueError("当前提交缺少有效独立核验。")
    task1_digest=sha256(pub/"submissions/latest/task1.csv")
    old_record=(pub/"docs/Kaggle测评记录.md").read_text(encoding="utf-8")
    paths=[
        "task2/tfidf_sparse.py","task2/sparse_noise.py","task2/english_prepare.py",
        "task2/english_infer.py","task2/english_audit.py","task2/english_backend_audit.py",
        "task2/english_smooth.py","task2/english_model_source.json","task2/requirements-english.txt",
        "task2/round8_fusion.py","task2/round8_joint.py","task2/round8_predict.py",
        "task2/tests/test_tfidf.py","task2/tests/test_smoothing.py","task2/tests/test_joint8.py",
        "task2/docs/DeBERTa与量化原理说明.md","task2/docs/第八轮复现说明.md",
        "docs/Task2第八轮优化记录-20261114.md","docs/Task2第八轮优化结果-20261114.md",
        "tools/finish_task2_round8.py","tools/report_task2_round8.py","tools/sync_task2_round8.py",
        "tools/verify_smoothing_graph.py","tools/audit_task2_shift8.py"]
    for relative in paths:
        target=pub/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    # 所有旧Task2小证据仍保留，但让当前入口只显示第八轮的结果。
    latest=pub/"task2/latest";history=latest/"parent_round7";history.mkdir(exist_ok=False)
    for old in latest.iterdir():
        if old.is_file():shutil.move(str(old),str(history/old.name))
    aliases={"confirmation.json":"confirmation.json","frozen_selection.json":"frozen_selection.json",
        "selection_protocol.json":"selection_protocol.json","uncertainty.json":"paired_uncertainty.json",
        "inner_comparison.json":"inner_comparison.json","reproduced.timing.json":"reproduced.timing.json",
        "english_fp32_backend_audit.json":"english_fp32_backend_audit.json",
        "english_development.timing.json":"english_development.timing.json",
        "english_quantization_audit_reduced_rejected.json":"english_quantization_reduced_rejected.json",
        "english_quantization_audit_model_smooth_v2_int8.json":"english_quantization_smooth_rejected.json",
        "raw_word_feature_verification.json":"raw_word_feature_verification.json",
        "raw_joint_input_verification.json":"raw_joint_input_verification.json",
        "english_test.timing.json":"english_test.timing.json",
        "test_input_equivalence.json":"test_input_equivalence.json",
        "model/file_integrity.json":"model_file_integrity.json",
        "unlabeled_shift_audit.json":"unlabeled_shift_audit.json"}
    for original,name in aliases.items():shutil.copy2(OUT/original,latest/name)
    shutil.copy2(OUT/"tfidf_split/training.json",latest/"word_training.json")
    shutil.copy2(ROOT/"submissions/20261114_task2_round8/manifest.json",latest/"verification_manifest.json")
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    write_json(latest/"protocol.json",{"date":"2026-11-14","round":8,"previous_measured_best":.53719,
        "new_kaggle_score":None,"inner_fit":33437,"inner_check":8268,"confirmation":41342,
        "final_head_fit":83047,"word_fit_rows":7531189,"outer_validation_rows":153456,
        "historical_validation_reuse":True,"gpu_used":False,"frozen_encoders":True,
        "english_precision":"f32","english_used":choice["english"],"selection_candidates":24})
    (latest/"README.md").write_text("# 当前证据：第八轮\n\n"
        "protocol、selection_protocol、frozen_selection、inner_comparison、confirmation、paired_uncertainty、"
        "reproduced.timing和verification_manifest均为本轮。线上新分数未知，第七轮已测0.53719。\n"
        "英语数值审计包含拒用INT8及正式FP32运行时核验。parent_round7是旧阶段证据，不能当成本轮指标。\n"
        "Git不含完整Task2模型；取得自包含模型包后按task2/README开头预测。\n",encoding="utf-8")
    shutil.copy2(source,pub/"submissions/latest/task2.csv")
    manifest=json.loads((pub/"submissions/latest/manifest.json").read_text(encoding="utf-8"))
    manifest["snapshot_date"]="2026-11-14"
    method="TF-IDF、冻结BERT"+("/英语DeBERTa" if choice["english"] else "")+"与自主噪声头"
    manifest["tasks"]["task2"]={**proof,"source_local_path":"submissions/20261114_task2_round8/task2_round8.csv",
        "submitted_file":"submissions/latest/task2.csv","model_included":False,"tests_passed":41,
        "separate_onnx_graph_test_passed":1,"latest_kaggle_score":None,"previous_measured_best":.53719,
        "method":method,"choice":choice}
    write_json(pub/"submissions/latest/manifest.json",manifest)
    title="# Task2当前入口：第八轮（2026-11-14）\n\n"
    title+="最新提交：submissions/latest/task2.csv。第七轮已测0.53719，本轮线上待测；本地风险不是Kaggle分数。\n\n"
    title+="方案、实际选择和消融见根目录docs/Task2第八轮优化结果-20261114.md；"
    title+="中文复现见docs/第八轮复现说明.md，理解材料见docs/DeBERTa与量化原理说明.md。\n\n"
    title+="Git不含大型权重。向维护者取得本轮整个model文件夹，放到task2/latest/model。\n\n"
    title+="```powershell\npython -m task2.round8_predict --model task2/latest/model --test comp-5434-20261-project-task-2/test.csv --semantic-python .venv-semantic/Scripts/python.exe --english-python .venv-english/Scripts/python.exe --work-dir task2_inference_round8 --output task2_prediction_round8.csv\npython -m unittest discover -s task2/tests -v\n```\n\n"
    title+="新增英语环境使用Python3.12和requirements-english.txt；数值和旧BERT环境仍独立。\n\n下方为旧阶段说明，当前入口以上文为准。\n\n---\n\n"
    task_readme=pub/"task2/README.md";task_readme.write_text(title+task_readme.read_text(encoding="utf-8"),encoding="utf-8")
    readme=pub/"README.md";text=readme.read_text(encoding="utf-8");lines=text.splitlines()
    positions=[i for i,line in enumerate(lines) if line.startswith("| Task 2 |")]
    if len(positions)!=1:raise RuntimeError("首页Task2入口变化，需要保留式合并。")
    lines[positions[0]]=f"| Task 2 | {method}，第八轮 | [task2.csv](submissions/latest/task2.csv) | 第八轮待测；第七轮已测RMSE=0.53719 |"
    text="\n".join(lines)+"\n"
    text=text.replace("Task 2本地第七轮的噪声风险估计","Task 2第八轮的噪声风险估计")
    text=text.replace("[本地第七轮结果](docs/Task2本地第七轮优化结果-20261114.md)","[第八轮结果](docs/Task2第八轮优化结果-20261114.md)")
    size="本轮约1.2GB的完整模型包" if choice["english"] else "数百MB的完整模型包"
    text=text.replace("数百MB的完整模型包",size)
    readme.write_text(text,encoding="utf-8")
    # 本地还存在其他Task1实验；只把Task2指定小节接到仓库原记录，避免带入无关文档链接。
    local=(ROOT/"docs/Kaggle测评记录.md").read_text(encoding="utf-8")
    marker="## 2026-11-14：第七轮反馈与第八轮"
    addition=local[local.index(marker):].strip()
    (pub/"docs/Kaggle测评记录.md").write_text(old_record.rstrip()+"\n\n"+addition+"\n",encoding="utf-8")
    upload=pub/"docs/最新上传说明.md"
    upload.write_text("# 当前更新：Task2第八轮\n\n"
        "Task2最新CSV、中文阶段记录、代码、41项回归和独立原始CSV预测证据已更新。第七轮已测0.53719，新分数待反馈。\n"
        "英语CPU运行时使用原FP32；失败INT8不发布为可用模型。Task2完整模型包需另取，Git不含权重或训练数据。\n"
        "Task1继续沿用仓库原文件；下方为历史上传说明。\n\n---\n\n"+upload.read_text(encoding="utf-8"),encoding="utf-8")
    write_json(pub/"docs/latest_validation.json",{"date":"2026-11-14","round":8,"task1_unchanged":True,
        "task2_tests_passed":41,"publication_source_tests_passed":None,"separate_graph_test_passed":1,
        "independent_raw_prediction_byte_identical":proof["independent_byte_identical"],
        "old_submission_files_verified":proof["old_files_unchanged"],"new_kaggle_score":None,"previous_measured_best":.53719})
    if sha256(pub/"submissions/latest/task1.csv")!=task1_digest:raise RuntimeError("Task1指纹发生变化。")
    print("第八轮发布副本已整理，等待在副本中核验后提交。",proof["sha256"],flush=True)


if __name__=="__main__":main()
