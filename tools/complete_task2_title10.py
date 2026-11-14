"""完成独立8候选标题分支；未过门槛保留第九轮，不自动推送或提交Kaggle。"""
from datetime import datetime,timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import task2
import numpy as np
import pandas as pd
from task2.audit import ROOT,CSV_OPTIONS,sha256,write_json
from task2.title10 import OUT,FOLDER,PARENT,CACHE
from tools.finish_task2_local7 import uncertainty


def main(resume_after_parent=False):
    ledger=OUT/"completion.json"
    if ledger.exists() and not resume_after_parent:raise FileExistsError("标题流水线已经存在，不覆盖现场。")
    home=Path.home()/".cache";cpu=home/"comp5434-english/venv/Scripts/python.exe";bert=home/"comp5434-round5/venv/Scripts/python.exe"
    numeric=Path(sys.executable);state={"status":"waiting_encoding","steps":[],"new_kaggle_score":None}
    if resume_after_parent:
        previous=json.loads(ledger.read_text(encoding="utf-8"))
        if previous["status"]!="failed" or (FOLDER/"model").exists():raise RuntimeError("只允许恢复父轮核验导致的打包前等待中断。")
        shutil.copy2(ledger,OUT/"completion_first_attempt.json")
        state={**previous,"status":"resuming_after_parent","previous_error":previous.get("error")};state.pop("error",None)
    journal=ROOT/"docs/Task2第十轮优化记录-20261114.md"
    def note(value):
        with journal.open("a",encoding="utf-8") as stream:stream.write("\n"+value+"\n")
    def module(py,name,*args):return [py,"-X","utf8","-m",name,*args]
    def run(label,args):
        row={"name":label,"started_utc":datetime.now(timezone.utc).isoformat()};state["steps"].append(row);write_json(ledger,state)
        print("标题流水线开始",label,flush=True);began=time.perf_counter()
        result=subprocess.run(list(map(str,args)),cwd=ROOT);row.update(seconds=time.perf_counter()-began,returncode=result.returncode)
        write_json(ledger,state);note(f"### {label}\n\n实际耗时{row['seconds']:.2f}秒，返回码{result.returncode}，开始UTC={row['started_utc']}。")
        if result.returncode:raise RuntimeError(label+"失败")
    def wait_for(condition,label,timeout=3600):
        began=time.monotonic();last=-1
        while not condition():
            parent=json.loads((PARENT/"completion_pipeline.json").read_text(encoding="utf-8"))
            if parent["status"]=="failed":raise RuntimeError("第九轮父流水线失败，停止标题分支。")
            elapsed=int(time.monotonic()-began)
            if elapsed>timeout:raise TimeoutError(label+"超时")
            if elapsed//60!=last:print(label,elapsed,"秒",flush=True);last=elapsed//60
            time.sleep(2)
    def reject(reason):
        state.update(status="rejected",reason=reason);write_json(ledger,state)
        write_json(OUT/"decision.json",{"accepted":False,"reason":reason,"recommended_round":9,"new_kaggle_score":None})
        note("本分支未采用："+reason+"。不根据确认结果重新选择其他候选，继续第九轮交付。")
    try:
        write_json(ledger,state)
        if not resume_after_parent:
            wait_for(lambda:(OUT/"encoding_timing.json").exists(),"等待冻结标题编码")
            state["status"]="running";write_json(ledger,state)
            timing=json.loads((OUT/"encoding_timing.json").read_text(encoding="utf-8"))
            if sha256(FOLDER/"development.npz")!=timing["logits_sha256"]:raise ValueError("标题输出指纹不符。")
            note("冻结标题编码已完成："+json.dumps(timing,ensure_ascii=False)+"。数值核验见gpu_fidelity.json。")
            run("8项内部标题对照",module(numeric,"task2.title10","--stage","select"))
            choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
            if not choice["internal_improved"]:reject("内部检查未优于第九轮");return
            run("冻结标题配置确认",module(numeric,"task2.title10","--stage","confirm"))
        else:
            choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
            audit=json.loads((OUT/"final_title_deployment_audit.json").read_text(encoding="utf-8"))
            if not choice["internal_improved"] or not audit["accepted"] or not (OUT/"final_head.npz").exists():raise RuntimeError("恢复时已有冻结头证据不完整。")
            note("父轮严格核验因线程舍入恢复；标题选择、确认与最终头均保持原样，跳过已经完成的监督阶段。")
        proof=json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
        if proof["candidate"]["noise_model_risk"]["latent_mse_estimate"]>=proof["baseline"]["noise_model_risk"]["latent_mse_estimate"]:
            reject("固定确认未优于第九轮");return
        note("标题分支确认风险RMSE："+str(proof["baseline"]["noise_model_risk"]["latent_rmse_estimate"])+" → "+str(proof["candidate"]["noise_model_risk"]["latent_rmse_estimate"])+"。配置保持冻结，线上分数未知。")
        if not resume_after_parent:run("83047行标题残差头",module(numeric,"task2.title10","--stage","fit-final"))
        # 等正文CPU流程完成再使用CPU全速编码标题，避免两套OpenVINO争用四个物理核心。
        wait_for(lambda:json.loads((PARENT/"completion_pipeline.json").read_text(encoding="utf-8"))["status"]=="complete","等待第九轮独立核验")
        old_files={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/"submissions").rglob("*.csv")};write_json(OUT/"old_submissions.json",old_files)
        model=FOLDER/"model";model.mkdir(exist_ok=False)
        def link_or_copy(source,target):
            try:os.link(source,target)
            except OSError:shutil.copy2(source,target)
            return target
        # 父包只读，硬链接为独立目录项；复制给队友时按普通文件读取，不依赖旧路径。
        shutil.copytree(CACHE/"model",model/"previous_model",copy_function=link_or_copy)
        shutil.copy2(OUT/"final_head.npz",model/"head.npz")
        write_json(model/"model.json",{"schema_version":1,"round":10,"choice":choice,"parent_round":9,"title_encoder_updated":False,
            "title_source_revision":"06f020e0dbf909570eb886423bd3af256b855546","title_max_length":128,
            "previous_measured_best":.50646,"new_kaggle_score":None,"historical_validation_reuse":True,"external_corpus_overlap":"unknown"})
        shutil.copy2(ROOT/"task2/docs/第十轮标题语义与复现.md",model/"UNDERSTANDING.md")
        write_json(model/"file_integrity.json",{str(p.relative_to(model)).replace("\\","/"):{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(model.rglob("*")) if p.is_file()})
        write_json(OUT/"model_reference.json",{"path":str(model),"logical_bytes":sum(p.stat().st_size for p in model.rglob("*") if p.is_file()),"parameters_in_git":False})
        english=model/"previous_model/previous_model/english"
        run("测试标题编码或受核验冻结缓存",module(cpu,"task2.english_infer","--model",english/"model.onnx","--tokenizer",english/"tokenizer",
            "--input",FOLDER/"test_title.csv","--output",FOLDER/"test.npz","--backend","openvino","--threads","4","--batch-size","16"))
        run("标题候选CSV生成",module(numeric,"task2.title10","--stage","export"))
        work=FOLDER/"independent";work.mkdir(exist_ok=False)
        previous=CACHE/"independent"
        for name in ["input.csv","embedding.npz","embedding.embedding.npy","embedding.logits.npy","embedding.state.json",
                     "english.npz","english.embedding.npy","english.logits.npy","english.state.json"]:shutil.copy2(previous/name,work/name)
        dest=work/"prefix";dest.mkdir()
        for name in ["tokens.json","tokens.npy","lengths.npy","offsets.npy","id.npy","hidden.npy","state.json"]:
            if name=="hidden.npy":link_or_copy(previous/"prefix"/name,dest/name)
            else:shutil.copy2(previous/"prefix"/name,dest/name)
        for suffix in [".npz",".embedding.npy",".logits.npy",".state.json"]:shutil.copy2(FOLDER/("test"+suffix),work/("title"+suffix))
        run("原始CSV标题/正文独立预测",module(numeric,"task2.round10_predict","--model",model,"--test",ROOT/"comp-5434-20261-project-task-2/test.csv",
            "--semantic-python",bert,"--english-python",cpu,"--work-dir",work,"--output",OUT/"reproduced.csv"))
        a=pd.read_csv(OUT/"submission.csv");b=pd.read_csv(OUT/"reproduced.csv");sample=pd.read_csv(ROOT/"comp-5434-20261-project-task-2/sampleSubmission.csv")
        if list(a.columns)!=["id","rating"] or len(a)!=10000 or not a.id.is_unique or not np.isfinite(a.rating).all() or not a.rating.between(1,5).all():raise ValueError("标题提交格式错误。")
        np.testing.assert_array_equal(a.id,sample.id);np.testing.assert_array_equal(a.id,b.id);np.testing.assert_allclose(a.rating,b.rating,atol=1e-8,rtol=0)
        parent=np.load(OUT/"reproduced.parent.npz");old=pd.read_csv(PARENT/"submission.csv")
        np.testing.assert_array_equal(parent["id"],old.id);np.testing.assert_allclose(parent["prediction"],old.rating,atol=6e-10,rtol=0)
        if not all(sha256(ROOT/p)==h for p,h in old_files.items()):raise ValueError("历史提交指纹变化。")
        target=ROOT/"submissions/20261114_task2_round10";target.mkdir(exist_ok=False);shutil.copy2(OUT/"submission.csv",target/"task2_round10.csv")
        result={"round":10,"rows":len(a),"columns":["id","rating"],"sha256":sha256(target/"task2_round10.csv"),"sample_id_order_exact":True,
            "independent_raw_feature_path":True,"independent_byte_identical":(OUT/"submission.csv").read_bytes()==(OUT/"reproduced.csv").read_bytes(),
            "max_abs_difference":float(np.abs(a.rating-b.rating).max()),"round9_parent_reproduced":True,"old_files_unchanged":len(old_files),
            "previous_measured_best":.50646,"new_kaggle_score":None,"model":str(model)}
        write_json(target/"manifest.json",result);write_json(OUT/"uncertainty.json",uncertainty(pd.read_csv(OUT/"confirmation_predictions.csv"),.25165131324376977))
        write_json(OUT/"decision.json",{"accepted":True,"recommended_round":10,"new_kaggle_score":None})
        state["status"]="complete";write_json(ledger,state);note("第十轮格式及独立复现通过："+json.dumps(result,ensure_ascii=False)+"。线上分数未知，等待Kaggle测评。")
        print("标题分支交付通过",result,flush=True)
    except Exception as error:
        state.update(status="failed",error=str(error));write_json(ledger,state);note("标题流水线停止："+str(error)+"。保留第九轮及所有现场。");raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--resume-after-parent",action="store_true")
    main(parser.parse_args().resume_after_parent)
