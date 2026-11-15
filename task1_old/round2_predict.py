"""从独立专家模型包生成 Task 1 整数预测，不依赖旧输出目录。"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import Preprocessor
from .boosting import HistogramBoosting
from .models import model_from_dict
from .core_experiment import predict
from .mixture import ConditionalMixture


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--test",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    frame=pd.read_csv(args.test)
    state=json.loads((args.model/"model.json").read_text(encoding="utf-8"))
    prior=json.loads((args.model/"previous_model/model.json").read_text(encoding="utf-8"))
    forest=json.loads((args.model/"previous_model/baseline_forest.json").read_text(encoding="utf-8"))
    old=HistogramBoosting.from_dict(prior["model"]).predict_proba(Preprocessor.from_dict(prior["preprocessor"]).transform(frame))
    bp=model_from_dict(forest["model"]).predict_proba(Preprocessor.from_dict(forest["preprocessor"]).transform(frame))
    old=(1-prior["baseline_weight"])*old+prior["baseline_weight"]*bp
    if state.get("family")=="mixture":
        mixture=ConditionalMixture.from_dict(state["model"]).predict_proba(frame)
        p=state["previous_weight"]*old+(1-state["previous_weight"])*mixture
    else:
        p=predict(frame,state["support"],Preprocessor.from_dict(state["preprocessor"]),
                  np.asarray(state["active"],dtype=bool),model_from_dict(state["model"]),old,state["previous_weight"])
    pd.DataFrame({"id":frame.id,"label":p.argmax(1)+1}).to_csv(args.output,index=False)


if __name__=="__main__":main()
