"""第十轮独立原始CSV预测：父轮正文/历史与冻结标题语义融合。

完整包内已含英语原网络，标题无需另一套外部权重。所有最终评分重算；
冻结向量断点仅在输入/模型/分词器/运行时指纹一致时复用，不读训练缓存。
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from .audit import ROOT,CSV_OPTIONS,sha256,write_json
from .round9_predict import predict as parent_predict,verify_model
from .partial9_fusion import design
from .noise_softmax import predict_channel


def predict(frame,bert,bert_cls,english,english_cls,adapted,title,title_cls,model):
    parent,structure,offset,known=parent_predict(frame,bert,bert_cls,english,english_cls,adapted,model/"previous_model",return_components=True)
    np.testing.assert_array_equal(title["id"],frame.id.to_numpy())
    meta=json.loads((model/"model.json").read_text(encoding="utf-8"))
    if meta["schema_version"]!=1 or meta["round"]!=10:raise ValueError("未知标题融合模型格式。")
    choice=meta["choice"]
    features=design(structure,parent,title["logits"],title_cls if choice["route"]=="cls_head" else None)
    changed=predict_channel(features,offset,dict(np.load(model/"head.npz")))
    result=np.clip(parent+choice["alpha"]*(changed-parent),1,5)
    return np.where(known,result,parent),parent


def main():
    p=argparse.ArgumentParser()
    for key in ["model","test","semantic-python","english-python","work-dir","output"]:p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.work_dir.resolve()==args.model.resolve():raise ValueError("工作目录不能与模型目录相同。")
    began=time.perf_counter();verify_model(args.model);args.work_dir.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(args.test,**CSV_OPTIONS)
    if not frame.id.is_unique:raise ValueError("测试ID重复。")
    text=args.work_dir/"input.csv";frame[["id","title","comment"]].fillna("").to_csv(text,index=False)
    titles=args.work_dir/"title_input.csv"
    pd.DataFrame({"id":frame.id,"title":"","comment":frame.title.fillna("").astype(str)}).to_csv(titles,index=False)
    parent9=args.model/"previous_model";parent8=parent9/"previous_model";parent6=parent8/"previous_model/previous_model"
    bert=args.work_dir/"embedding.npz";english=args.work_dir/"english.npz";adapted=args.work_dir/"adapted.npz";title=args.work_dir/"title.npz"
    def run(python,module,*rest):subprocess.run([str(python),"-X","utf8","-m",module,*map(str,rest)],cwd=ROOT,check=True)
    run(args.semantic_python,"task2.embedding_infer","--encoder",parent6/"encoder.onnx","--tokenizer",parent6/"tokenizer.json","--input",text,"--output",bert)
    for input_path,output_path in [(text,english),(titles,title)]:
        run(args.english_python,"task2.english_infer","--model",parent8/"english/model.onnx","--tokenizer",parent8/"english/tokenizer",
            "--input",input_path,"--output",output_path,"--backend","openvino","--threads","4","--batch-size","16")
    run(args.english_python,"task2.partial9_infer","--model",parent9/"adapter","--input",text,"--cache-folder",args.work_dir/"prefix","--output",adapted)
    values=[dict(np.load(path)) for path in [bert,english,adapted,title]]
    for value in values:np.testing.assert_array_equal(value["id"],frame.id.to_numpy())
    result,parent=predict(frame,values[0],np.load(bert.with_suffix(".embedding.npy"),mmap_mode="r"),values[1],
        np.load(english.with_suffix(".embedding.npy"),mmap_mode="r"),values[2],values[3],np.load(title.with_suffix(".embedding.npy"),mmap_mode="r"),args.model)
    if not np.isfinite(result).all():raise ValueError("最终评分非有限。")
    pd.DataFrame({"id":frame.id,"rating":result}).to_csv(args.output,index=False,float_format="%.10f")
    np.savez_compressed(args.output.with_suffix(".parent.npz"),id=frame.id.to_numpy(),prediction=parent)
    write_json(args.output.with_suffix(".timing.json"),{"rows":len(frame),"seconds":time.perf_counter()-began,"sha256":sha256(args.output),
        "title_rows_resumed":json.loads(title.with_suffix(".timing.json").read_text(encoding="utf-8"))["rows_resumed"],
        "prefix_rows_resumed":json.loads(adapted.with_suffix(".timing.json").read_text(encoding="utf-8"))["prefix_rows_resumed"],
        "top_threads":json.loads(adapted.with_suffix(".timing.json").read_text(encoding="utf-8")).get("top_threads",1),"training_cache_required":False})


if __name__=="__main__":main()
