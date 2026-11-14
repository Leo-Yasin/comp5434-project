"""第八轮自包含预测：原始CSV+模型包，无须训练缓存或任何历史提交文件。

旧BERT和新英语网络分别在固定环境推理，断点需验证网络/分词器/输入指纹。
词模型与TF-IDF归一化在本进程从原始CSV重建，不能把缓存中的最终分数当输入。
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import ROOT,CSV_OPTIONS,sha256,write_json
from .local7_predict import raw_features
from .features import build_text,TEXT_DIM
from .lookup_context import context_from_frame
from .tfidf_sparse import transform
from .semantic_fusion import semantic_features
from .embedding_heads import linear_logits
from .noise_softmax import softmax,predict_channel
from . import round8_joint as joint


def word_inputs(frame,parent6):
    """恢复旧模型包中的原尺度float32历史特征，避免从舍入过的组合特征反推。"""
    previous4=parent6/"support/previous_model"
    previous3=previous4/json.loads((previous4/"model.json").read_text(encoding="utf-8"))["prior"]
    previous2=previous3/json.loads((previous3/"model.json").read_text(encoding="utf-8"))["support"]
    base=previous2/"base_model"
    eps=json.loads((base/"model.json").read_text(encoding="utf-8"))["score_calibration"]["epsilon"]
    numeric=context_from_frame(frame,base);numeric[:,[0,3,6]]*=1-eps
    return build_text(frame),numeric.astype(np.float32)


def predict(frame,semantic,embedding,model,english=None,english_embedding=None,return_components=False):
    description=json.loads((model/"model.json").read_text(encoding="utf-8"));choice=description["choice"]
    if description["schema_version"]!=1:raise ValueError("未知第八轮模型格式。")
    parent7=model/"previous_model";parent6=parent7/"previous_model"
    old,x6,parent6_score,offset6,known=raw_features(frame,semantic,embedding,parent6)
    text,numeric=word_inputs(frame,parent6)
    previous_word=np.load(parent7/"word.npz")
    old_logits=text@previous_word["weights"][:TEXT_DIM]+numeric@previous_word["weights"][TEXT_DIM:]+previous_word["bias"]
    x7=np.column_stack([x6[:,:60],parent6_score,semantic_features(old_logits,parent6_score)]).astype(np.float32)
    offset7=linear_logits(x7,offset6,dict(np.load(parent7/"head.npz")))
    parent7_score=softmax(offset7)@np.arange(1.,6.)
    parent7_score=np.where(known,parent7_score,parent6_score)
    blocks=[np.column_stack([x7,parent7_score])]
    lexical_logits=None
    if choice["lexical"]:
        new_word=np.load(model/"word.npz");idf=np.load(model/"idf.npy")
        weighted=transform(text,idf)
        logits=weighted@new_word["weights"][:TEXT_DIM]+numeric@new_word["weights"][TEXT_DIM:]+new_word["bias"]
        lexical_logits=logits
        blocks.append(semantic_features(logits,parent7_score))
    if choice["english"]:
        if english is None or english_embedding is None:raise ValueError("缺少英语编码结果。")
        np.testing.assert_array_equal(english["id"],frame.id.to_numpy())
        blocks.extend([semantic_features(english["logits"],parent7_score),np.asarray(english_embedding)])
    if choice.get("family","residual").startswith("joint_"):
        if lexical_logits is None or english is None or old.shape[1]<39:raise ValueError("联合头缺少完整独立输入。")
        features,new_offset,_=joint.assemble(old[:,:35],old[:,36],semantic["logits"],embedding,
            lexical_logits,english["logits"],english_embedding)
        changed=joint.clean_prediction(features,new_offset,dict(np.load(model/"head.npz")),choice)
    else:
        features=np.column_stack(blocks).astype(np.float32)
        changed=predict_channel(features,offset7,dict(np.load(model/"head.npz")))
    result=np.clip(parent7_score+choice["alpha"]*(changed-parent7_score),1,5)
    result=np.clip(np.where(known,result,parent7_score),1,5)
    if return_components:
        # 第九轮需要相应第八轮的完整类别偏置，不能从一个连续评分反推5个logits。
        # 默认返回保持原行为；当前第八轮包明确为residual且alpha=1。
        if choice.get("family","residual")!="residual" or choice["alpha"]!=1.:
            raise ValueError("该第八轮配置没有可直接继承的噪声类别偏置。")
        offset8=linear_logits(features,offset7,dict(np.load(model/"head.npz")))
        return result,old[:,:35],offset8,known
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ["model","test","semantic-python","work-dir","output"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--english-python",type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.model.resolve()==args.work_dir.resolve():raise ValueError("模型与工作目录不能相同。")
    args.work_dir.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    integrity=json.loads((args.model/"file_integrity.json").read_text(encoding="utf-8"))
    for relative,expected in integrity.items():
        target=args.model/relative
        # manifest来自本次打包，但仍要求路径留在模型目录内，防止错误引用外部旧文件。
        if args.model.resolve() not in target.resolve().parents:raise ValueError("模型清单路径越界。")
        if not target.is_file() or target.stat().st_size!=expected["bytes"] or sha256(target)!=expected["sha256"]:
            raise ValueError("模型文件缺失或指纹不符："+relative)
    description=json.loads((args.model/"model.json").read_text(encoding="utf-8"));choice=description["choice"]
    frame=pd.read_csv(args.test,**CSV_OPTIONS);text=args.work_dir/"input.csv"
    frame[["id","title","comment"]].fillna("").to_csv(text,index=False)
    parent6=args.model/"previous_model/previous_model"
    parent_meta=json.loads((parent6/"model.json").read_text(encoding="utf-8"))
    if sha256(parent6/"encoder.onnx")!=parent_meta["encoder_sha256"]:raise ValueError("旧编码器指纹不符。")
    output=args.work_dir/"embedding.npz"
    subprocess.run([str(args.semantic_python),"-X","utf8","-m","task2.embedding_infer","--encoder",str(parent6/"encoder.onnx"),
        "--tokenizer",str(parent6/"tokenizer.json"),"--input",str(text),"--output",str(output)],check=True,cwd=ROOT)
    semantic=dict(np.load(output));embedding=np.load(output.with_suffix(".embedding.npy"),mmap_mode="r")
    np.testing.assert_array_equal(semantic["id"],frame.id.to_numpy());english=None;english_embedding=None
    if choice["english"]:
        if args.english_python is None:raise ValueError("所选模型需要英语CPU推理环境。")
        if sha256(args.model/"english/model.onnx")!=description["english_sha256"]:raise ValueError("英语编码器指纹不符。")
        target=args.work_dir/"english.npz";runtime=description["english_runtime"]
        subprocess.run([str(args.english_python),"-X","utf8","-m","task2.english_infer","--model",str(args.model/"english/model.onnx"),
            "--tokenizer",str(args.model/"english/tokenizer"),"--input",str(text),"--output",str(target),
            "--threads",str(runtime["threads"]),"--batch-size",str(runtime["batch_size"]),"--backend",runtime["backend"]],check=True,cwd=ROOT)
        english=dict(np.load(target));english_embedding=np.load(target.with_suffix(".embedding.npy"),mmap_mode="r")
    result=predict(frame,semantic,embedding,args.model,english,english_embedding)
    if not np.isfinite(result).all() or not frame.id.is_unique:raise ValueError("最终预测/ID非法。")
    pd.DataFrame({"id":frame.id,"rating":result}).to_csv(args.output,index=False,float_format="%.10f")
    timings={"seconds":time.perf_counter()-began,"rows":len(frame),"sha256":sha256(args.output),"training_cache_required":False,
        "old_encoder_rows_resumed":json.loads(output.with_suffix(".timing.json").read_text(encoding="utf-8"))["rows_resumed"]}
    if choice["english"]:timings["english_rows_resumed"]=json.loads(target.with_suffix(".timing.json").read_text(encoding="utf-8"))["rows_resumed"]
    write_json(args.output.with_suffix(".timing.json"),timings)


if __name__=="__main__":main()
