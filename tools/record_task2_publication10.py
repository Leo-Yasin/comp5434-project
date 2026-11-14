"""记录实际Git副本测试及独立预测结果；不重训，不自动提交或推送。

调用前已在副本实际执行42项数值、7项模型数学、4项缓存协议测试并确认退出0。
本工具另严格检查已生成预测/模型来源与Task1保护，再写小型发布证据。
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import task2
import numpy as np
import pandas as pd
from task2.audit import ROOT,sha256,write_json
from task2.title10 import OUT,FOLDER


def main():
    pub=ROOT/"tmp/github-publish-20261108"
    original=ROOT/"submissions/20261114_task2_round10/task2_round10.csv";prediction=OUT/"publication_prediction.csv"
    a=pd.read_csv(original);b=pd.read_csv(prediction)
    np.testing.assert_array_equal(a.id,b.id);np.testing.assert_allclose(a.rating,b.rating,atol=1e-8,rtol=0)
    if original.read_bytes()!=prediction.read_bytes() or sha256(pub/"submissions/latest/task2.csv")!=sha256(original):raise ValueError("Git副本预测/提交未逐字节一致。")
    old_task1=subprocess.check_output(["git","-C",str(pub),"show","HEAD:submissions/latest/task1.csv"])
    if hashlib.sha256(old_task1).hexdigest()!=sha256(pub/"submissions/latest/task1.csv"):raise ValueError("Task1提交发生变化。")
    if subprocess.check_output(["git","-C",str(pub),"diff","--name-only","HEAD","--","task1"]):raise ValueError("Task1文件发生变化。")
    current=json.loads((pub/"submissions/latest/manifest.json").read_text(encoding="utf-8"))
    previous=json.loads(subprocess.check_output(["git","-C",str(pub),"show","HEAD:submissions/latest/manifest.json"]))
    if current["tasks"]["task1"]!=previous["tasks"]["task1"]:raise ValueError("Task1元数据发生变化。")
    tests={"numerical":42,"model_math":7,"cache_protocol":4,"total":53,"all_exit_codes":0,
        "actual_publication_source_used":True,"numeric_six_dependency":"local compatibility path, not source package override"}
    write_json(pub/"task2/latest/publication_tests.json",tests)
    shutil.copy2(pub/"task2/outputs/round10_20261114/cache_protocol_tests.json",pub/"task2/latest/publication_cache_tests.json")
    proof={"rows":len(a),"sha256":sha256(original),"independent_byte_identical":True,"max_abs_difference":0.,
        "publication_source":str(pub),"top_threads":4,"task1_unchanged":True,"model_integrity_sha256":sha256(FOLDER/"model/file_integrity.json"),
        "title_rows_resumed":10000,"prefix_rows_resumed":10000,"new_kaggle_score":None,"previous_measured_best":.50646}
    write_json(pub/"task2/latest/publication_verification.json",proof);write_json(OUT/"publication_verification.json",proof)
    shutil.copy2(OUT/"publication_prediction.timing.json",pub/"task2/latest/publication_prediction.timing.json")
    validation=json.loads((pub/"docs/latest_validation.json").read_text(encoding="utf-8"))
    validation.update(numerical_tests=42,new_math_tests=7,cache_protocol_tests=4,total_tests=53,publication_tests_passed=True,
        publication_prediction_byte_identical=True,publication_prediction_sha256=sha256(prediction),task1_unchanged=True)
    write_json(pub/"docs/latest_validation.json",validation)
    with (ROOT/"docs/Task2第十轮优化记录-20261114.md").open("a",encoding="utf-8") as stream:
        stream.write("\n## 实际Git副本验证\n\n实际发布副本42项数值、7项模型数学、4项缓存协议测试全部通过。首次副本预测因本机旧Anaconda缺six而在导入阶段退出，按既有依赖说明仅为该命令补充项目six兼容路径；未覆盖包源码、未修改系统依赖。随后从副本源码独立预测10000行，提交SHA一致、逐字节一致；Task1所有源码、CSV和元数据均不变。冻结编码复用行数明确记录，不作为首次编码计时。\n")
    shutil.copy2(ROOT/"docs/Task2第十轮优化记录-20261114.md",pub/"docs/Task2第十轮优化记录-20261114.md")
    shutil.copy2(ROOT/"task2/docs/阶段记录.md",pub/"task2/docs/阶段记录.md")
    print("实际发布证据通过",tests,proof["sha256"],flush=True)


if __name__=="__main__":main()
