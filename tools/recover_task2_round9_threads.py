"""保留单线程原产物，把正式两条预测路径统一为包内四线程后严格复核。

32条256样本逐位一致不代表所有动态长度都一致；首次全量差2.958e-7。
不放宽1e-8门槛，不重训、不改冻结选择。CPU前缀复用，最后层及评分重算。
"""
from datetime import datetime,timezone
import json
import shutil
import subprocess
import sys
import task2
import numpy as np
import pandas as pd
from task2.audit import ROOT,write_json
from task2.partial9_protocol import OUT,FOLDER


def main():
    ledger=OUT/"completion_pipeline.json";state=json.loads(ledger.read_text(encoding="utf-8"))
    if state["status"]!="failed" or (ROOT/"submissions/20261114_task2_round9").exists():raise RuntimeError("恢复入口不符合原首次核验失败现场。")
    shutil.copy2(ledger,OUT/"completion_first_attempt.json")
    a=pd.read_csv(OUT/"submission.csv");b=pd.read_csv(OUT/"reproduced.csv")
    np.testing.assert_array_equal(a.id,b.id)
    difference=float(np.abs(a.rating-b.rating).max())
    write_json(OUT/"cpu_threads_full_test_audit.json",{"rows":len(a),"single_vs_four_max_rating_error":difference,
        "bit_identical":False,"original_tolerance":1e-8,"original_passed":False,"resolution":"canonical four-thread inference on both independent paths"})
    if difference>1e-5:raise RuntimeError("误差超过预期线程舍入范围，拒绝自动处理。")
    for old,new in [("neural_test.npz","neural_test_single_thread.npz"),("neural_test.timing.json","neural_test_single_thread.timing.json"),("submission.csv","submission_single_thread.csv")]:
        target=OUT/new
        if target.exists():raise FileExistsError(target)
        shutil.copy2(OUT/old,target)
    shutil.copy2(FOLDER/"test_cpu/timing.json",OUT/"test_prefix_fresh_timing.json")
    state.update(status="running",previous_error=state.pop("error",None),recovery="canonical CPU threads; no training/configuration changes")
    write_json(ledger,state)
    journal=ROOT/"docs/Task2第九轮优化记录-20261114.md"
    with journal.open("a",encoding="utf-8") as stream:stream.write(f"\n## 全量线程差异与严格复核恢复\n\n原单线程与四线程在10000行上最大差{difference}，原1e-8门槛未通过。32条256样本逐位相同不覆盖所有动态形状。保留原单线程NN/CSV和首次失败记录；统一包内四线程，重新计算正式NN/CSV后与原始CSV独立四线程路径复核。未放宽门槛、未改模型/选择。首次完整CPU前缀计时另存test_prefix_fresh_timing.json，恢复时的断点计时不冒称首次编码耗时。\n")
    cpu=FOLDER.parent/"comp5434-english/venv/Scripts/python.exe"
    jobs=[("统一四线程重新预测",[cpu,"-X","utf8","-m","task2.partial9_infer","--model",FOLDER/"model/adapter","--input",ROOT/"task2/outputs/round6_20261108/test_text.csv","--cache-folder",FOLDER/"test_cpu","--output",OUT/"neural_test.npz"]),
        ("重新生成正式CSV",[sys.executable,"-X","utf8","-m","task2.partial9_fusion","--stage","final"]),
        ("原严格核验",[sys.executable,"-X","utf8","-m","tools.finish_task2_round9","--stage","verify"]),
        ("实际结果报告",[sys.executable,"-X","utf8","-m","tools.report_task2_round9"])]
    for label,command in jobs:
        print("线程恢复",label,flush=True)
        row={"name":label,"started_utc":datetime.now(timezone.utc).isoformat()};state.setdefault("recovery_steps",[]).append(row);write_json(ledger,state)
        result=subprocess.run(list(map(str,command)),cwd=ROOT);row["returncode"]=result.returncode;write_json(ledger,state)
        if result.returncode:state.update(status="failed",error=label+"失败");write_json(ledger,state);raise RuntimeError(label)
    state["status"]="complete";write_json(ledger,state)
    with journal.open("a",encoding="utf-8") as stream:stream.write("\n统一四线程后的原始CSV独立预测与格式核验通过；以manifest的实际最大差和逐字节标记为准。\n")
    print("第九轮统一线程严格复核完成",flush=True)


if __name__=="__main__":main()
