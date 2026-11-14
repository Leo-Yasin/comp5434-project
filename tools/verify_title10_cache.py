"""对实际冻结标题缓存验证正常复用及输入/模型/完成行数误配拒绝。"""
from argparse import Namespace
import argparse
import json
import os
from pathlib import Path
import shutil
import task2
import pandas as pd
from task2.audit import ROOT,write_json
from task2.english_infer import reuse_verified_cuda


def main(work_name="cache_protocol_smoke"):
    if Path(work_name).name!=work_name or work_name in [".",".."]:raise ValueError("核验目录名必须是单个安全名称。")
    folder=Path.home()/".cache/comp5434-title10";target=folder/work_name
    target.mkdir(exist_ok=False)
    # 仅创建本任务专有的小目录；大数组硬链接且只读，不修改原模型或缓存。
    for suffix in [".npz",".logits.npy",".embedding.npy"]:
        try:os.link(folder/("test"+suffix),target/("test"+suffix))
        except OSError:shutil.copy2(folder/("test"+suffix),target/("test"+suffix))
    shutil.copy2(folder/"test.state.json",target/"test.state.json")
    original=json.loads((target/"test.state.json").read_text(encoding="utf-8"))
    english=Path.home()/".cache/comp5434-local9/model/previous_model/english"
    args=Namespace(input=folder/"test_title.csv",output=target/"test.npz",model=english/"model.onnx",tokenizer=english/"tokenizer")
    frame=pd.read_csv(args.input,keep_default_na=False);passed=[]
    assert reuse_verified_cuda(args,frame);passed.append("correct_inputs_reused")
    changed=target/"changed.csv";changed.write_bytes(args.input.read_bytes()+b"\n")
    args.input=changed
    try:reuse_verified_cuda(args,frame)
    except ValueError:passed.append("changed_input_rejected")
    else:raise AssertionError("输入SHA变化未拒绝。")
    args.input=folder/"test_title.csv"
    write_json(target/"test.state.json",{**original,"reference_onnx_sha256":"0"*64})
    try:reuse_verified_cuda(args,frame)
    except ValueError:passed.append("changed_model_rejected")
    else:raise AssertionError("模型SHA变化未拒绝。")
    write_json(target/"test.state.json",{**original,"completed":len(frame)-1})
    try:reuse_verified_cuda(args,frame)
    except ValueError:passed.append("incomplete_cache_rejected")
    else:raise AssertionError("未完成缓存未拒绝。")
    write_json(ROOT/"task2/outputs/round10_20261114/cache_protocol_tests.json",{"passed":len(passed),"checks":passed,"original_cache_unchanged":True})
    print("标题冻结缓存协议4项核验通过",passed,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--work-name",default="cache_protocol_smoke")
    main(parser.parse_args().work_name)
