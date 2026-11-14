"""本地第七轮自包含预测：原始CSV、冻结语义网络及新的本地参数。

不读取训练缓存或旧提交CSV。继承的第四/五/六轮参数随包保留，新增历史表
仅在请求身份有支持时生效；新五星词模型通过与训练一致的数值特征计算logits。
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import ROOT,CSV_OPTIONS,sha256,write_json
from .semantic_predict import inherited_from_frame
from .embedding_features import combine
from .embedding_heads import predict_head,ridge_predict
from .regression_boost import ResidualHistogramBoost
from .semantic_fusion import known_history_mask,semantic_features
from .posterior_history import history_design,FIELDS,lookup_sums
from .sparse_fusion import old_logits
from .noise_softmax import softmax,predict_channel
from .features import build_text,TEXT_DIM
from .lookup_context import context_from_frame


def raw_features(frame,semantic,embedding,parent_model):
    """复现旧六轮的完整输入，和缓存路径比较时须保持相同ONNX批组成。"""
    description=json.loads((parent_model/"model.json").read_text(encoding="utf-8"))
    support=parent_model/"support"
    old_features,old_score=inherited_from_frame(frame,support/"previous_model")
    linear=dict(np.load(support/"parent_linear.npz"))
    booster=ResidualHistogramBoost.from_dict(json.loads((support/"parent_booster.json").read_text(encoding="utf-8")))
    features,parent5=combine(old_features,old_score,semantic["logits"],embedding,linear,booster,description["parent_alpha"])
    offset=old_logits(features,semantic["logits"],dict(np.load(parent_model/"head.npz")))
    parent6=softmax(offset)@np.arange(1.,6.)
    known=known_history_mask(old_features)
    parent6=np.where(known,parent6,parent5)
    return old_features,features,parent6,offset,known


def predict(frame,semantic,embedding,model):
    description=json.loads((model/"model.json").read_text(encoding="utf-8"))
    if description["schema_version"]!=1 or description["variant"] not in ["history","sparse"]:
        raise ValueError("未知的本地第七轮模型包。")
    parent_model=model/"previous_model"
    old,features,parent,offset,known=raw_features(frame,semantic,embedding,parent_model)
    choice=description["choice"]
    if description["variant"]=="history":
        queries={field:pd.util.hash_pandas_object(frame[field].fillna("").astype(str),index=False).to_numpy(dtype=np.uint64) for field in FIELDS}
        states={field:dict(np.load(model/f"{field}.npz")) for field in FIELDS}
        new_features=history_design(queries,states,parent)
        x=np.column_stack([features[:,:60],parent,new_features]).astype(np.float32)
        if choice["family"]=="ridge":changed=ridge_predict(x,parent,dict(np.load(model/"head.npz")))
        else:
            booster=ResidualHistogramBoost.from_dict(json.loads((model/"head.json").read_text(encoding="utf-8")))
            changed=booster.predict(x,parent,choice["rounds"])
        # 任一身份缺少所请求表的统计时，原预测仍可独立处理任意新CSV。
        usable=known&np.all(np.column_stack([lookup_sums(queries[field],states[field])[:,0]>0 for field in FIELDS]),axis=1)
    else:
        previous4=parent_model/"support/previous_model"
        previous3=previous4/json.loads((previous4/"model.json").read_text(encoding="utf-8"))["prior"]
        previous2=previous3/json.loads((previous3/"model.json").read_text(encoding="utf-8"))["support"]
        base=previous2/"base_model"
        eps=json.loads((base/"model.json").read_text(encoding="utf-8"))["score_calibration"]["epsilon"]
        numeric=context_from_frame(frame,base);numeric[:,[0,3,6]]*=1-eps
        numeric=numeric.astype(np.float32)
        word=dict(np.load(model/"word.npz"));text=build_text(frame)
        logits=text@word["weights"][:TEXT_DIM]+numeric@word["weights"][TEXT_DIM:]+word["bias"]
        x=np.column_stack([features[:,:60],parent,semantic_features(logits,parent)]).astype(np.float32)
        changed=predict_channel(x,offset,dict(np.load(model/"head.npz")))
        usable=known
    result=np.clip(parent+choice["alpha"]*(changed-parent),1,5)
    return np.where(usable,result,parent)


def main():
    parser=argparse.ArgumentParser()
    for name in ["model","test","semantic-python","work-dir","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.work_dir.resolve()==args.model.resolve():raise ValueError("工作目录不能是模型目录。")
    started=time.perf_counter();args.work_dir.mkdir(parents=True,exist_ok=True)
    parent=args.model/"previous_model";frame=pd.read_csv(args.test,**CSV_OPTIONS)
    parent_description=json.loads((parent/"model.json").read_text(encoding="utf-8"))
    if sha256(parent/"encoder.onnx")!=parent_description["encoder_sha256"]:
        raise ValueError("继承编码器指纹不匹配。")
    frame[["id","title","comment"]].fillna("").to_csv(args.work_dir/"input.csv",index=False)
    output=args.work_dir/"embedding.npz"
    subprocess.run([str(args.semantic_python),"-X","utf8","-m","task2.embedding_infer","--encoder",str(parent/"encoder.onnx"),
        "--tokenizer",str(parent/"tokenizer.json"),"--input",str(args.work_dir/"input.csv"),"--output",str(output)],check=True,cwd=ROOT)
    semantic=dict(np.load(output));embedding=np.load(output.with_suffix(".embedding.npy"),mmap_mode="r")
    np.testing.assert_array_equal(semantic["id"],frame.id.to_numpy())
    result=predict(frame,semantic,embedding,args.model)
    if not np.isfinite(result).all() or not frame.id.is_unique:raise ValueError("输出评分或ID非法。")
    pd.DataFrame({"id":frame.id,"rating":result}).to_csv(args.output,index=False,float_format="%.10f")
    timing=json.loads(output.with_suffix(".timing.json").read_text(encoding="utf-8"))
    write_json(args.output.with_suffix(".timing.json"),{"seconds":time.perf_counter()-started,"rows":len(frame),
        "sha256":sha256(args.output),"independent_raw_csv":True,"training_cache_required":False,
        "semantic_rows_resumed":timing["rows_resumed"],"encoder_sha256":parent_description["encoder_sha256"]})


if __name__=="__main__":main()
