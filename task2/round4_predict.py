"""第四轮独立推理：原模型包、可选稳健词参数和新增残差树。

稳健词评分仅作辅助特征，不擅自套用旧噪声校正当成最终评分。
所有旧校正/新残差只在各自定义的位置应用一次。
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import CSV_OPTIONS,write_json,sha256
from .features import build_text,TEXT_DIM
from .round3_features import features_from_frame
from .lookup_model import residual_design
from .regression_boost import ResidualHistogramBoost
from .round4_features import refinement_matrix
from .round4_train import old_predict


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--test",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    began=time.perf_counter()
    description=json.loads((args.model/"model.json").read_text(encoding="utf-8"));chosen=description["chosen"]
    if description["schema_version"]!=1:raise ValueError("不支持的模型版本。")
    prior_model=args.model/description["prior"]
    prior_description=json.loads((prior_model/"model.json").read_text(encoding="utf-8"))
    support=prior_model/prior_description["support"]
    epsilon=json.loads((support/"model.json").read_text(encoding="utf-8"))["epsilon_full"]
    frame=pd.read_csv(args.test,**CSV_OPTIONS)
    x,base,means,counts,context=features_from_frame(frame,support)
    coefficient=np.load(prior_model/"linear.npz")["coefficient"]
    initial=base+residual_design(base,means,counts,context)@coefficient
    inherited_model=ResidualHistogramBoost.from_dict(json.loads((prior_model/"booster.json").read_text(encoding="utf-8")))
    prior=old_predict(x,initial,inherited_model,prior_description["alpha"])
    auxiliary=np.zeros(len(frame))
    if chosen["use_auxiliary"]:
        state=np.load(args.model/"auxiliary.npz");w=state["weights"]
        numeric=context.copy();numeric[:,[0,3,6]]*=(1-epsilon);numeric=numeric.astype(np.float32)
        text=build_text(frame)
        auxiliary=float(state["mean"])+text@w[:TEXT_DIM]+numeric@w[TEXT_DIM:]
    features=refinement_matrix(x,prior,auxiliary,epsilon,chosen["use_auxiliary"])
    booster=ResidualHistogramBoost.from_dict(json.loads((args.model/"booster.json").read_text(encoding="utf-8")))
    p=booster.predict(features,prior);p=np.clip(prior+chosen["alpha"]*(p-prior),1,5)
    seconds=time.perf_counter()-began
    if not np.isfinite(p).all() or not frame.id.is_unique:raise ValueError("评分或ID不合法。")
    pd.DataFrame({"id":frame.id,"rating":p}).to_csv(args.output,index=False,float_format="%.10f")
    write_json(args.output.with_suffix(".timing.json"),{"rows":len(frame),"complete_inference_seconds":seconds,
        "includes_loading_reading_all_features":True,"sha256":sha256(args.output)})
    print(f"第四轮独立推理 {len(frame):,} 条，耗时 {seconds:.3f}s",flush=True)


if __name__=="__main__":main()
