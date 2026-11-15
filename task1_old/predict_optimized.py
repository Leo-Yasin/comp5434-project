"""从优化模型目录独立预测，支持已保存的提升树/基线概率融合。

示例：python -m task1.predict_optimized --model task1/outputs/optimization_20261105
      --test comp-5434-20261-project-task-1/test.csv --output task1/outputs/optimization_20261105/reproduced.csv
"""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from .data import Preprocessor
from .boosting import HistogramBoosting
from .models import model_from_dict


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--test",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    state=json.loads((args.model/"model.json").read_text(encoding="utf-8"));frame=pd.read_csv(args.test)
    prep=Preprocessor.from_dict(state["preprocessor"])
    if state["model"].get("type")=="forest":
        # 森林调参候选保存了训练折内选择的活动列；推理时只能复用这些列。
        p=model_from_dict(state["model"]).predict_proba(prep.transform(frame)[:,np.asarray(state["active"],dtype=bool)])
        w=0.0
    else:
        p=HistogramBoosting.from_dict(state["model"]).predict_proba(prep.transform(frame))
        w=state["baseline_weight"]
    if w:
        base=json.loads((args.model/"baseline_forest.json").read_text(encoding="utf-8"))
        bp=model_from_dict(base["model"]).predict_proba(Preprocessor.from_dict(base["preprocessor"]).transform(frame))
        p=(1-w)*p+w*bp
    pd.DataFrame({"id":frame.id,"label":p.argmax(1)+1}).to_csv(args.output,index=False)


if __name__=="__main__":main()
