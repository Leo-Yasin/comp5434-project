"""第八轮打包/校验：只发布通过内部与冻结确认的候选，不自动向Kaggle提交。"""
import argparse
import json
import shutil
from pathlib import Path
import task2
import numpy as np
import pandas as pd
from task2.audit import ROOT,sha256,write_json
from task2.round8_fusion import OUT
from tools.finish_task2_local7 import uncertainty


def pack():
    if not (OUT/"submission.csv").exists():raise FileNotFoundError("等待最终预测导出完成。")
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    result=json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
    if not choice["internal_improved"] or result["candidate"]["noise_model_risk"]["latent_mse_estimate"]>=result["baseline"]["noise_model_risk"]["latent_mse_estimate"]:
        raise RuntimeError("第八轮未通过，不发布。")
    model=OUT/"model";model.mkdir(exist_ok=False)
    write_json(OUT/"old_submissions.json",{str(p.relative_to(ROOT)).replace("\\","/"):sha256(p) for p in (ROOT/"submissions").rglob("*.csv")})
    shutil.copytree(ROOT/"task2/outputs/round7_local_20261114/model_sparse",model/"previous_model")
    shutil.copy2(OUT/"final_head.npz",model/"head.npz")
    description={"schema_version":1,"choice":choice,"previous_measured_best_kaggle":.53719,"new_kaggle_score":None,
        "word_fit_rows":7531189,"word_outer_validation_excluded":153456,"word_optimizer_not_claimed_converged":True,
        "historical_validation_reuse":True,"teacher_allows_pretrained_user_report":True,"cold_identity_fallback":"round7"}
    if choice["lexical"]:
        state=OUT/"tfidf_split"/(f"checkpoint_{choice['lexical']}.npz" if choice["lexical"]!=40 else "weights.npz")
        shutil.copy2(state,model/"word.npz")
        config=json.loads((OUT/"tfidf_split/training.json").read_text(encoding="utf-8"))["representation"]
        shutil.copy2(Path(config["folder"])/"idf.npy",model/"idf.npy")
        description["word_representation"]="training-only IDF; separate title/body L2 normalization"
    if choice["english"]:
        source=Path.home()/".cache/comp5434-english/model";target=model/"english";target.mkdir()
        audit=json.loads((OUT/"english_fp32_backend_audit.json").read_text(encoding="utf-8"))
        if not audit["accepted"]:raise RuntimeError("英语FP32数值检查未通过。")
        shutil.copy2(source/"model.onnx",target/"model.onnx")
        shutil.copytree(source/"tokenizer_spm",target/"tokenizer")
        for source_name,target_name in [("README.md","SOURCE_MODEL_CARD.md"),("provenance.json","provenance.json"),
                                        ("config.json","source_config.json"),("export.json","export.json")]:
            shutil.copy2(source/source_name,target/target_name)
        shutil.copy2(ROOT/"task2/docs/DeBERTa与量化原理说明.md",model/"UNDERSTANDING.md")
        description.update(english_runtime={"backend":"openvino","threads":4,"batch_size":16,"max_length":128,"precision":"f32"},
            english_sha256=sha256(target/"model.onnx"),english_model_revision="06f020e0dbf909570eb886423bd3af256b855546",
            english_finetuned=False,english_quantization_used=False,external_corpus_overlap="unknown")
    write_json(model/"model.json",description)
    epsilon=float(np.load(ROOT/"task2/outputs/round3_20261105/development.npz")["epsilon"])
    write_json(OUT/"uncertainty.json",uncertainty(pd.read_csv(OUT/"confirmation_predictions.csv"),epsilon))
    (model/"MODEL_USAGE.md").write_text("# 第八轮完整模型\n\n使用task2.round8_predict，从原始CSV重建词、历史及语义特征。\n"
        "模型包自包含；无需训练缓存或历史提交CSV。\n若choice.english为true，使用独立英语环境的OpenVINO FP32，不能替换为已失败的INT8文件。\n"
        "BERT使用原ORT CPU环境，英语网络没有微调；中文原理材料在UNDERSTANDING.md及项目task2/docs中。\n"
        "线上新分数未知，本地噪声风险不是Kaggle分数。\n",encoding="utf-8")
    shutil.copy2(ROOT/"task2/docs/第八轮复现说明.md",model/"REPRODUCING.md")
    # 传给队友的是多层完整目录，逐文件指纹能发现漏拷贝、旧头混入或网络被替换。
    write_json(model/"file_integrity.json",{str(path.relative_to(model)).replace("\\","/"):
        {"bytes":path.stat().st_size,"sha256":sha256(path)} for path in sorted(model.rglob("*")) if path.is_file()})
    print("第八轮模型包就绪",model,flush=True)


def verify():
    source=OUT/"submission.csv";other=OUT/"reproduced.csv"
    first=pd.read_csv(source);second=pd.read_csv(other)
    sample=pd.read_csv(ROOT/"comp-5434-20261-project-task-2/sampleSubmission.csv")
    if list(first.columns)!=["id","rating"] or len(first)!=10000 or not first.id.is_unique:raise ValueError("提交格式非法。")
    np.testing.assert_array_equal(first.id,sample.id)
    if not np.isfinite(first.rating).all() or not first.rating.between(1,5).all():raise ValueError("提交评分非法。")
    np.testing.assert_allclose(first.rating,second.rating,atol=1e-8,rtol=0)
    old=json.loads((OUT/"old_submissions.json").read_text(encoding="utf-8"))
    if not all(sha256(ROOT/p)==h for p,h in old.items()):raise ValueError("旧提交指纹变化。")
    folder=ROOT/"submissions/20261114_task2_round8";folder.mkdir(exist_ok=False)
    target=folder/"task2_round8.csv";shutil.copy2(source,target)
    proof={"rows":10000,"sha256":sha256(target),"columns":["id","rating"],"sample_id_order_exact":True,
        "independent_raw_feature_path":True,"independent_byte_identical":source.read_bytes()==other.read_bytes(),
        "max_abs_difference":float(np.abs(first.rating-second.rating).max()),"old_files_unchanged":len(old),
        "new_kaggle_score":None,"previous_measured_best":.53719,"model":"task2/outputs/round8_20261114/model"}
    write_json(folder/"manifest.json",proof)
    print("第八轮候选通过校验",str(target),proof,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["pack","verify"],required=True);args=p.parse_args()
    globals()[args.stage]()
