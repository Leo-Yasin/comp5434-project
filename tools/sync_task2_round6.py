"""把已完成校验的第六轮整理到现有Git副本；不推送、不复制大型模型。

保留Task1和仓库已有文件，只更新Task2当前代码/资料与唯一latest CSV。
模型实际原件始终保留在训练工作区，历史候选由本地目录与Git历史共同保留。
"""
import json
import shutil
import subprocess
from pathlib import Path

from task2.audit import ROOT,sha256,write_json


def main():
    pub=ROOT/"tmp/github-publish-20261108"
    if subprocess.check_output(["git","-C",str(pub),"status","--porcelain"]):
        raise RuntimeError("发布副本有未提交修改，不能覆盖。")
    if subprocess.check_output(["git","-C",str(pub),"remote","get-url","origin"],text=True).strip()!="https://github.com/Leo-Yasin/comp5434-project.git":
        raise RuntimeError("发布目标仓库不匹配。")
    candidate=ROOT/"submissions/20261108_task2_round6"
    proof=json.loads((candidate/"manifest.json").read_text(encoding="utf-8"))
    source=candidate/"task2_cls_adapted.csv"
    if sha256(source)!=proof["submission_sha256"] or not proof["independent_raw_csv_prediction_byte_identical"]:
        raise RuntimeError("新候选缺少完整验证或指纹不一致。")
    def copy(relative):
        target=pub/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    for file in (ROOT/"task2").glob("embedding_*.py"):copy(file.relative_to(ROOT))
    for relative in [".gitignore","task2/tests/test_embedding.py","task2/requirements-embedding.txt",
        "task2/docs/阶段记录.md","task2/docs/BERT原理与本项目答辩指南.md","docs/Kaggle测评记录.md",
        "docs/Task2第六轮优化记录-20261108.md","docs/Task2第六轮优化结果-20261108.md",
        "tools/finish_task2_round6.py","tools/sync_task2_round6.py"]:
        copy(relative)
    header="""# GitHub最新入口：Task2第六轮（2026-11-08）

当前候选为冻结BERT的768维CLS+自主噪声线性锚点+64维MLP残差头，最新CSV为 `../submissions/latest/task2.csv`。
新线上分数待反馈；此前第五轮已测RMSE=0.54604。老师允许外部预训练模型（用户转述），前提是理解模型。
理解材料见 `docs/BERT原理与本项目答辩指南.md`，具体结果见根目录第六轮报告。

本仓库只包含源码、配置和验证证据，完整模型包仍需维护者单独提供，解压为 `task2/latest/model`。
原数值Python按requirements.txt安装；独立Python3.12语义环境按requirements-embedding.txt安装。
取得模型和原始测试CSV后，在项目根目录运行（语义Python路径按本机替换）：

```powershell
python -m task2.embedding_predict --model task2/latest/model --test comp-5434-20261-project-task-2/test.csv --semantic-python .venv-semantic/Scripts/python.exe --work-dir task2_inference_round6 --output task2_prediction_round6.csv
python -m unittest discover -s task2/tests -v
```

下面保留历轮开发说明；其中outputs路径是当时本地训练产物，未随Git上传。

---

"""
    (pub/"task2/README.md").write_text(header+(ROOT/"task2/README.md").read_text(encoding="utf-8"),encoding="utf-8")
    shutil.copy2(source,pub/"submissions/latest/task2.csv")
    manifest=json.loads((pub/"submissions/latest/manifest.json").read_text(encoding="utf-8"))
    manifest["tasks"]["task2"]={"source_local_path":str(source.relative_to(ROOT)).replace("\\","/"),
        "rows":10000,"sha256":proof["submission_sha256"],"submitted_file":"submissions/latest/task2.csv",
        "method":"frozen BERT CLS 768 + custom forward-noise linear anchor and MLP residual",
        "model_included":False,"latest_kaggle_score":None,"previous_measured_best":.54604,
        "external_model_allowed_teacher_confirmation_user_report":True,"condition":"understand pretrained model",
        "tests_passed":proof["tests_passed"]}
    write_json(pub/"submissions/latest/manifest.json",manifest)
    out=ROOT/"task2/outputs/round6_20261108"
    for name in ["protocol.json","frozen_selection.json","confirmation.json","paired_uncertainty.json",
        "reproduced.timing.json","test_parent_verification.json","linear_anchor_confirmation_read_only.json"]:
        shutil.copy2(out/name,pub/"task2/latest"/name)
    shutil.copy2(candidate/"manifest.json",pub/"task2/latest/verification_manifest.json")
    # 仅移除latest目录中上一轮专属的单个审计副本，原始历史文件不动，Git历史也保留。
    obsolete=pub/"task2/latest/extra_audit.json"
    if obsolete.exists():obsolete.unlink()
    readme=pub/"README.md";text=readme.read_text(encoding="utf-8")
    replacements={
        "冻结BERT上下文语义、评分噪声校正与自主非线性融合，第五轮最终候选":"冻结BERT的CLS768维与自主噪声MLP任务头，第六轮候选",
        "待用户反馈；此前第四轮已测RMSE=0.59347":"第六轮待反馈；此前第五轮已测RMSE=0.54604",
        "Task 2第五轮的本地噪声风险估计不等于Kaggle成绩；使用外部预训练模型，比赛是否允许尚未核对。":
        "Task 2第六轮的本地噪声风险估计不等于Kaggle成绩。用户转述老师允许外部预训练模型，条件是理解模型；已提供中文原理/答辩指南。",
        "[第五轮模型说明](task2/docs/第五轮模型说明.md)、[第五轮结果](docs/Task2第五轮优化结果-20261105.md)":
        "[BERT原理与答辩指南](task2/docs/BERT原理与本项目答辩指南.md)、[第六轮结果](docs/Task2第六轮优化结果-20261108.md)"}
    for before,after in replacements.items():
        if text.count(before)!=1:raise RuntimeError("仓库首页已变化，需要人工保留并合并："+before)
        text=text.replace(before,after)
    readme.write_text(text,encoding="utf-8")
    boundary=pub/"docs/最新上传说明.md"
    boundary.write_text("# 更新：Task2第六轮（2026-11-08）\n\nTask1保持原样，Task2的latest CSV更新为第六轮CLS+噪声MLP。\n"
        "第五轮已测0.54604，新版线上待反馈；新代码、理解材料、29项测试和原始CSV独立复现证据已同步。\n"
        "完整Task2参数包仍需另行获取，解压为task2/latest/model；GitHub不含原始数据或大型模型。\n\n"
        "以下为首次上传时的历史范围说明，当前版本以首页和latest/manifest为准。\n\n---\n\n"+
        boundary.read_text(encoding="utf-8"),encoding="utf-8")
    write_json(pub/"docs/latest_validation.json",{"date":"2026-11-08","task1_unchanged":True,
        "task2_unit_tests_passed":proof["tests_passed"],"independent_raw_csv_prediction_byte_identical":True,
        "all_old_submissions_unchanged":True,"task2_model_not_in_git":True,"new_task2_kaggle_score":None})
    print("第六轮发布副本准备完成；未执行推送。",proof["submission_sha256"],flush=True)


if __name__=="__main__":main()
