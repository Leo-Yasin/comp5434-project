"""从第三轮独立模型包预测评分，不读取训练集或缓存。

原词模型已完成噪声校正，新树学习的是潜在评分残差，不能再对最终输出重复校正。
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import CSV_OPTIONS,write_json,sha256
from .round3_features import features_from_frame,FEATURE_NAMES
from .lookup_model import residual_design
from .regression_boost import ResidualHistogramBoost


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--test",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    began=time.perf_counter()
    description=json.loads((args.model/"model.json").read_text(encoding="utf-8"))
    if description["schema_version"]!=1 or description["features"]!=FEATURE_NAMES:raise ValueError("特征定义或模型版本不匹配。")
    frame=pd.read_csv(args.test,**CSV_OPTIONS)
    x,base,means,counts,context=features_from_frame(frame,args.model/description["support"])
    coefficient=np.load(args.model/"linear.npz")["coefficient"]
    initial=base+residual_design(base,means,counts,context)@coefficient
    booster=ResidualHistogramBoost.from_dict(json.loads((args.model/"booster.json").read_text(encoding="utf-8")))
    p=booster.predict(x,initial);p=np.clip(initial+description["alpha"]*(p-initial),1,5)
    seconds=time.perf_counter()-began
    if not np.isfinite(p).all() or not frame.id.is_unique:raise ValueError("预测或 id 不合法。")
    pd.DataFrame({"id":frame.id,"rating":p}).to_csv(args.output,index=False,float_format="%.10f")
    write_json(args.output.with_suffix(".timing.json"),{"rows":len(frame),"complete_inference_seconds":seconds,
        "includes_model_loading_csv_features_predictions":True,"sha256":sha256(args.output)})
    print(f"第三轮独立推理 {len(frame):,} 条，耗时 {seconds:.3f}s",flush=True)


if __name__=="__main__":main()
