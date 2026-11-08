"""第六轮原始CSV独立推理：单次冻结网络得到logits/CLS，再重建第五轮基线和新头。

只需要自包含模型包、原始测试CSV和两个Python环境，不需要任何训练缓存。
新头只对历史支持条件满足的记录生效，缺历史输入回退第五轮评分。
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd

from .audit import ROOT,CSV_OPTIONS,write_json,sha256
from .semantic_predict import inherited_from_frame
from .semantic_fusion import known_history_mask
from .regression_boost import ResidualHistogramBoost
from .embedding_features import combine
from .embedding_heads import predict_head


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True);parser.add_argument("--test",type=Path,required=True)
    parser.add_argument("--semantic-python",type=Path,required=True);parser.add_argument("--work-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.work_dir.resolve()==args.model.resolve():raise ValueError("推理工作目录不能是模型目录。")
    began=time.perf_counter();args.work_dir.mkdir(parents=True,exist_ok=True)
    description=json.loads((args.model/"model.json").read_text(encoding="utf-8"))
    if description["schema_version"]!=1 or sha256(args.model/"encoder.onnx")!=description["encoder_sha256"]:
        raise ValueError("模型版本或编码网络指纹不符。")
    frame=pd.read_csv(args.test,**CSV_OPTIONS)
    text=args.work_dir/"input_text.csv";semantic=args.work_dir/"embedding.npz"
    frame[["id","title","comment"]].fillna("").to_csv(text,index=False)
    subprocess.run([str(args.semantic_python.resolve()),"-X","utf8","-m","task2.embedding_infer",
        "--encoder",str((args.model/"encoder.onnx").resolve()),"--tokenizer",str((args.model/"tokenizer.json").resolve()),
        "--input",str(text.resolve()),"--output",str(semantic.resolve())],check=True,cwd=ROOT)
    data=np.load(semantic);embedding=np.load(semantic.with_suffix(".embedding.npy"),mmap_mode="r")
    if not np.array_equal(data["id"],frame.id.to_numpy()):raise ValueError("原始测试与CLS的id错位。")
    support=args.model/"support"
    old_features,old_score=inherited_from_frame(frame,support/"previous_model")
    linear=dict(np.load(support/"parent_linear.npz"))
    booster=ResidualHistogramBoost.from_dict(json.loads((support/"parent_booster.json").read_text(encoding="utf-8")))
    features,parent=combine(old_features,old_score,data["logits"],embedding,linear,booster,description["parent_alpha"])
    prediction=predict_head(features,parent,data["logits"],dict(np.load(args.model/"head.npz")),description["choice"])
    known=known_history_mask(old_features);prediction=np.where(known,prediction,parent)
    if not np.isfinite(prediction).all() or not frame.id.is_unique:raise ValueError("最终评分或id非法。")
    pd.DataFrame({"id":frame.id,"rating":prediction}).to_csv(args.output,index=False,float_format="%.10f")
    timing=json.loads(semantic.with_suffix(".timing.json").read_text(encoding="utf-8"))
    write_json(args.output.with_suffix(".timing.json"),{"rows":len(frame),"complete_inference_seconds":time.perf_counter()-began,
        "semantic_checkpoint_reused":timing["rows_resumed"]>0,"sha256":sha256(args.output),
        "known_identity_rows":int(known.sum()),"includes_raw_csv_all_features_and_neural_encoder":True})
    print("第六轮完整独立预测完成",str(args.output),flush=True)


if __name__=="__main__":main()
