"""从独立第二轮模型包重新预测，不读取训练评分、验证标签或训练缓存。

先由包内第一轮校正模型计算基预测，再查询文本/交互统计、计算收缩残差。
推理计时包括两层模型加载、读取输入、构造特征与所有预测，不含最终写 CSV。
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import CSV_OPTIONS,sha256,write_json
from .predict import predict_file
from .lookup_features import make_keys,KEYS
from .lookup_model import lookup,predict_residual


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--test",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    began=time.perf_counter()
    description=json.loads((args.model/"model.json").read_text(encoding="utf-8"))
    if description["schema_version"]!=1 or description["keys"]!=KEYS:raise ValueError("不支持的模型格式。")
    base_file=args.output.with_suffix(".base.csv")
    predict_file(args.model/description["base"],args.test,base_file)
    base=pd.read_csv(base_file)
    frame=pd.read_csv(args.test,**CSV_OPTIONS)
    keys=make_keys(frame);means=np.empty((len(frame),len(KEYS)));counts=np.empty_like(means)
    for j,key in enumerate(KEYS):
        with np.load(args.model/f"{key}.npz") as state:
            means[:,j],counts[:,j]=lookup(keys[key],state,keys.get("valid_"+key))
    coefficient=np.load(args.model/"residual.npz")["coefficient"]
    if description.get("use_context",False):
        from .lookup_context import context_from_frame
        context=context_from_frame(frame,args.model/description["base"])
    else:context=None
    prediction=predict_residual(base.rating.to_numpy(),means,counts,coefficient,context=context)
    seconds=time.perf_counter()-began
    pd.DataFrame({"id":frame.id,"rating":prediction}).to_csv(args.output,index=False,float_format="%.10f")
    write_json(args.output.with_suffix(".timing.json"),{"rows":len(frame),
        "complete_inference_seconds":seconds,"sha256":sha256(args.output)})
    print(f"第二轮独立推理完成 {len(frame):,} 条，耗时 {seconds:.3f}s",flush=True)


if __name__=="__main__":main()
