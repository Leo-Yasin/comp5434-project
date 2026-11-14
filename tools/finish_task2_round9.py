"""只交付通过冻结确认的第九轮；模型放C盘，旧模型/CSV保持原样。

pack在最终适配头/校准头完成后准备自包含模型，verify在原始CSV预测完成后
核查评分、格式、旧文件指纹，最后创建新的推荐目录。无Kaggle提交操作。
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import task2
import numpy as np
import pandas as pd
from task2.audit import ROOT,sha256,write_json
from task2.partial9_protocol import FOLDER,OUT,tag,run_name
from tools.finish_task2_local7 import uncertainty


def pack():
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    proof=json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
    if not choice["internal_improved"] or proof["candidate"]["noise_model_risk"]["latent_mse_estimate"]>=proof["baseline"]["noise_model_risk"]["latent_mse_estimate"]:
        raise RuntimeError("第九轮未通过，不发布。")
    if not json.loads((OUT/"local_gpu_audit.json").read_text(encoding="utf-8"))["accepted"]:raise RuntimeError("GPU审计未通过。")
    if not json.loads((OUT/"adapted_deployment_audit.json").read_text(encoding="utf-8"))["accepted"]:
        raise RuntimeError("最终适配权重的CPU部署数值未通过。")
    if choice["max_length"]==256:
        for name in ["split_precision_audit_256.json","local_gpu_audit_256.json"]:
            if not json.loads((OUT/name).read_text(encoding="utf-8"))["accepted"]:raise RuntimeError("256token审计未通过。")
    top=FOLDER/run_name("final",choice)/f"epoch_{choice['epochs']}.pt"
    if not top.exists() or not (OUT/"final_fusion_training.json").exists():raise FileNotFoundError("等待最终适配和校准参数完成。")
    model=FOLDER/"model";model.mkdir(exist_ok=False)
    write_json(OUT/"old_submissions.json",{str(p.relative_to(ROOT)).replace("\\","/"):sha256(p) for p in (ROOT/"submissions").rglob("*.csv")})
    shutil.copytree(ROOT/"task2/outputs/round8_20261114/model",model/"previous_model")
    adapter=model/"adapter";adapter.mkdir()
    # 同卷硬链接是独立目录项；原路径删除后内容仍保留，跨机器复制时按普通文件读取。
    # 避免在空间有限的C盘再重复占用707MB；若文件系统不支持则退回完整复制。
    try:os.link(FOLDER/"prefix.onnx",adapter/"prefix.onnx")
    except OSError:shutil.copy2(FOLDER/"prefix.onnx",adapter/"prefix.onnx")
    shutil.copy2(top,adapter/"top.pt");shutil.copytree(FOLDER/"config",adapter/"config")
    shutil.copytree(model/"previous_model/english/tokenizer",adapter/"tokenizer")
    thread_audit=OUT/"cpu_threads_audit.json"
    threads=json.loads(thread_audit.read_text(encoding="utf-8"))["top_threads"] if thread_audit.exists() else 1
    write_json(adapter/"runtime.json",{"prefix_backend":"openvino","prefix_batch":16,"hidden_dtype":"float16",
        "top_backend":"torch_cpu","top_batch":16,"top_threads":threads,"max_length":choice["max_length"],"export_cls":choice["route"]=="cls_head"})
    if choice["route"]!="blend":shutil.copy2(OUT/"final_fusion.npz",model/"fusion.npz")
    write_json(model/"model.json",{"schema_version":1,"choice":choice,"previous_measured_best":.50646,"new_kaggle_score":None,
        "teacher_permits_pretrained_user_report":True,"frozen_encoder_layers":11,"adapted_encoder_layers":1,
        "adaptation_training_rows":83047,"training_max_length":128,"inference_max_length":choice["max_length"],
        "source_revision":"06f020e0dbf909570eb886423bd3af256b855546","external_corpus_overlap":"unknown",
        "historical_validation_reuse":True,"cold_identity_fallback":"round8"})
    shutil.copy2(ROOT/"task2/docs/第九轮最后层适配与复现.md",model/"UNDERSTANDING.md")
    write_json(model/"file_integrity.json",{str(p.relative_to(model)).replace("\\","/"):
        {"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(model.rglob("*")) if p.is_file()})
    write_json(OUT/"model_reference.json",{"path":str(model),"logical_bytes":sum(p.stat().st_size for p in model.rglob("*") if p.is_file()),
        "parameters_in_git":False,"reason":"F盘空间有限，完整模型存C盘本轮目录。"})
    epsilon=float(np.load(OUT/"controls.npz")["epsilon"])
    write_json(OUT/"uncertainty.json",uncertainty(pd.read_csv(OUT/"confirmation_predictions.csv"),epsilon))
    print("第九轮完整模型已准备",model,flush=True)


def verify():
    original=OUT/"submission.csv";reproduced=OUT/"reproduced.csv"
    a=pd.read_csv(original);b=pd.read_csv(reproduced)
    sample=pd.read_csv(ROOT/"comp-5434-20261-project-task-2/sampleSubmission.csv")
    if list(a.columns)!=["id","rating"] or len(a)!=10000 or not a.id.is_unique:raise ValueError("提交格式不正确。")
    np.testing.assert_array_equal(a.id,sample.id);np.testing.assert_array_equal(a.id,b.id)
    if not np.isfinite(a.rating).all() or not a.rating.between(1,5).all():raise ValueError("评分不合法。")
    np.testing.assert_allclose(a.rating,b.rating,atol=1e-8,rtol=0)
    parent=np.load(OUT/"reproduced.parent.npz")
    old=pd.read_csv(ROOT/"task2/outputs/round8_20261114/submission.csv")
    np.testing.assert_array_equal(parent["id"],old.id);np.testing.assert_allclose(parent["prediction"],old.rating,atol=6e-10,rtol=0)
    snapshot=json.loads((OUT/"old_submissions.json").read_text(encoding="utf-8"))
    if not all(sha256(ROOT/p)==h for p,h in snapshot.items()):raise ValueError("旧提交指纹变化。")
    folder=ROOT/"submissions/20261114_task2_round9";folder.mkdir(exist_ok=False)
    target=folder/"task2_round9.csv";shutil.copy2(original,target)
    proof={"rows":10000,"columns":["id","rating"],"sha256":sha256(target),"sample_id_order_exact":True,
        "independent_raw_feature_path":True,"independent_byte_identical":original.read_bytes()==reproduced.read_bytes(),
        "max_abs_difference":float(np.abs(a.rating-b.rating).max()),"round8_parent_reproduced":True,
        "old_files_unchanged":len(snapshot),"previous_measured_best":.50646,"new_kaggle_score":None,
        "model":str(FOLDER/"model")}
    write_json(folder/"manifest.json",proof);print("第九轮提交核验通过",proof,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["pack","verify"],required=True)
    args=p.parse_args();{"pack":pack,"verify":verify}[args.stage]()
