"""第九轮自包含预测：原始CSV、完整模型包和两个CPU语义环境。

冻结编码可在指纹一致时复用；最终类别、历史/词特征和评分始终重新计算。
训练用GPU不代表队友预测也必须有GPU，部署最后层使用固定CPU FP32实现。
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from .audit import ROOT,CSV_OPTIONS,sha256,write_json
from .round8_predict import predict as parent_predict
from .partial9_fusion import design
from .noise_softmax import predict_channel,softmax


def verify_model(model):
    for relative,expected in json.loads((model/"file_integrity.json").read_text(encoding="utf-8")).items():
        path=model/relative
        if model.resolve() not in path.resolve().parents:raise ValueError("模型路径越界。")
        if not path.is_file() or path.stat().st_size!=expected["bytes"] or sha256(path)!=expected["sha256"]:
            raise ValueError("模型缺失或指纹变化："+relative)


def predict(frame,bert,bert_cls,english,english_cls,adapted,model,adapted_cls=None,return_components=False):
    meta=json.loads((model/"model.json").read_text(encoding="utf-8"));choice=meta["choice"]
    if meta["schema_version"]!=1:raise ValueError("未知第九轮格式。")
    parent,structure,offset,known=parent_predict(frame,bert,bert_cls,model/"previous_model",english,english_cls,return_components=True)
    np.testing.assert_array_equal(adapted["id"],frame.id.to_numpy())
    logits=adapted["logits"]
    if choice.get("objective")=="residual_noise":logits=offset+logits-english["logits"]
    if choice["route"]=="blend":
        neural=softmax(logits)@np.arange(1.,6.)
        result=parent+choice["alpha"]*(neural-parent)
    else:
        if choice["route"]=="cls_head" and adapted_cls is None:raise ValueError("缺少适配CLS。")
        features=design(structure,parent,logits,adapted_cls if choice["route"]=="cls_head" else None)
        result=predict_channel(features,offset,dict(np.load(model/"fusion.npz")))
    result=np.clip(np.where(known,result,parent),1,5)
    if return_components:
        # 第十轮需继承真实的五星概率混合；默认返回与第九轮原CLI完全不变。
        if choice["route"]!="blend":raise ValueError("当前继承接口仅支持已冻结的概率期望混合。")
        from .title10 import mixture_offset
        offset9=mixture_offset(offset,logits,choice["alpha"])
        return result,structure,offset9,known
    return result,parent


def main():
    p=argparse.ArgumentParser()
    for key in ["model","test","semantic-python","english-python","work-dir","output"]:
        p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.work_dir.resolve()==args.model.resolve():raise ValueError("工作目录和模型不能相同。")
    began=time.perf_counter();verify_model(args.model);args.work_dir.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(args.test,**CSV_OPTIONS)
    if not frame.id.is_unique:raise ValueError("输入ID重复。")
    text=args.work_dir/"input.csv";frame[["id","title","comment"]].fillna("").to_csv(text,index=False)
    parent8=args.model/"previous_model";parent6=parent8/"previous_model/previous_model"
    bert=args.work_dir/"embedding.npz";english=args.work_dir/"english.npz";adapted=args.work_dir/"adapted.npz"
    subprocess.run([str(args.semantic_python),"-X","utf8","-m","task2.embedding_infer","--encoder",str(parent6/"encoder.onnx"),
        "--tokenizer",str(parent6/"tokenizer.json"),"--input",str(text),"--output",str(bert)],check=True,cwd=ROOT)
    subprocess.run([str(args.english_python),"-X","utf8","-m","task2.english_infer","--model",str(parent8/"english/model.onnx"),
        "--tokenizer",str(parent8/"english/tokenizer"),"--input",str(text),"--output",str(english),"--backend","openvino",
        "--threads","4","--batch-size","16"],check=True,cwd=ROOT)
    subprocess.run([str(args.english_python),"-X","utf8","-m","task2.partial9_infer","--model",str(args.model/"adapter"),
        "--input",str(text),"--cache-folder",str(args.work_dir/"prefix"),"--output",str(adapted)],check=True,cwd=ROOT)
    bert_values=dict(np.load(bert));english_values=dict(np.load(english));adapted_values=dict(np.load(adapted))
    for saved in [bert_values,english_values,adapted_values]:np.testing.assert_array_equal(saved["id"],frame.id.to_numpy())
    selected=json.loads((args.model/"model.json").read_text(encoding="utf-8"))["choice"]
    cls=np.load(adapted.with_suffix(".cls.npy"),mmap_mode="r") if selected["route"]=="cls_head" else None
    result,parent=predict(frame,bert_values,np.load(bert.with_suffix(".embedding.npy"),mmap_mode="r"),
        english_values,np.load(english.with_suffix(".embedding.npy"),mmap_mode="r"),adapted_values,args.model,cls)
    if not np.isfinite(result).all():raise ValueError("最终评分非有限。")
    pd.DataFrame({"id":frame.id,"rating":result}).to_csv(args.output,index=False,float_format="%.10f")
    timing={"rows":len(frame),"seconds":time.perf_counter()-began,"sha256":sha256(args.output),"training_cache_required":False,
        "bert_rows_resumed":json.loads(bert.with_suffix(".timing.json").read_text(encoding="utf-8"))["rows_resumed"],
        "english_rows_resumed":json.loads(english.with_suffix(".timing.json").read_text(encoding="utf-8"))["rows_resumed"],
        "prefix_rows_resumed":json.loads(adapted.with_suffix(".timing.json").read_text(encoding="utf-8"))["prefix_rows_resumed"],
        "top_threads":json.loads(adapted.with_suffix(".timing.json").read_text(encoding="utf-8")).get("top_threads",1)}
    write_json(args.output.with_suffix(".timing.json"),timing)
    np.savez_compressed(args.output.with_suffix(".parent.npz"),id=frame.id.to_numpy(),prediction=parent)


if __name__=="__main__":main()
