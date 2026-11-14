"""整理已通过本地确认的第七轮候选，保留旧提交并核验原始CSV独立预测。

只对通过内部分组和冻结确认的路线打包，不把未通过的尝试写成成功模型。
不访问Kaggle提交接口，不自动消耗比赛提交次数。
"""
import argparse
import json
import shutil
from pathlib import Path
import task2  # 先初始化原数值环境的可选six兼容路径，再导入pandas。
import numpy as np
import pandas as pd
from task2.audit import ROOT,sha256,write_json
from task2.history_prepare import OUT


def files(variant):
    if variant=="history":return OUT/"submission.csv",OUT/"confirmation.json",OUT/"frozen_selection.json"
    return OUT/"sparse_submission.csv",OUT/"sparse_confirmation.json",OUT/"sparse_frozen.json"


def uncertainty(frame,epsilon):
    """组级配对bootstrap，区间条件于噪声假设和已冻结模型，不消除历史选型偏差。"""
    y=frame.rating.to_numpy();g=frame.groups.to_numpy(dtype=np.uint64)
    unique,inverse=np.unique(g,return_inverse=True);counts=np.bincount(inverse)
    def risk(p):return ((p-y)**2-epsilon*(p*p-6*p+11))/(1-epsilon)
    old=risk(frame.baseline.to_numpy());new=risk(frame.prediction.to_numpy())
    a=np.bincount(inverse,weights=old);b=np.bincount(inverse,weights=new)
    rng=np.random.default_rng(5442);deltas=[]
    for _ in range(500):
        at=rng.integers(0,len(unique),len(unique));n=counts[at].sum()
        if min(a[at].sum(),b[at].sum())>0:deltas.append(np.sqrt(b[at].sum()/n)-np.sqrt(a[at].sum()/n))
    return {"difference":float(np.sqrt(new.mean())-np.sqrt(old.mean())),
        "conditional_95_interval":np.quantile(deltas,[.025,.975]).tolist(),"replicates":len(deltas),
        "limitations":"fixed uniform noise model; reused historical validation; external corpus overlap unknown; not Kaggle score"}


def pack(variant):
    source,confirmation,selection=files(variant)
    if not source.exists():raise FileNotFoundError("最终预测尚未完成，先等待训练/导出进程结束。")
    confirmed=json.loads(confirmation.read_text(encoding="utf-8"));choice=json.loads(selection.read_text(encoding="utf-8"))
    if not choice["internal_improved"] or confirmed["candidate"]["noise_model_risk"]["latent_mse_estimate"]>=confirmed["baseline"]["noise_model_risk"]["latent_mse_estimate"]:
        raise RuntimeError("内部与确认未共同通过，不能打包。")
    if not (OUT/"old_submissions.json").exists():
        snapshot={str(path.relative_to(ROOT)).replace("\\","/"):sha256(path) for path in (ROOT/"submissions").rglob("*.csv")}
        write_json(OUT/"old_submissions.json",snapshot)
    model=OUT/("model_"+variant);model.mkdir(exist_ok=False)
    shutil.copytree(ROOT/"task2/outputs/round6_20261108/model",model/"previous_model")
    if variant=="history":
        for field in ["user_id","prod_id","parent_prod_id"]:shutil.copy2(OUT/f"full.{field}.npz",model/f"{field}.npz")
        suffix=".npz" if choice["family"]=="ridge" else ".json"
        shutil.copy2(OUT/("final_head"+suffix),model/("head"+suffix))
    else:
        shutil.copy2(OUT/"sparse_final_head.npz",model/"head.npz")
        folder=OUT/"sparse_scaled_split"
        weights=folder/(f"checkpoint_{choice['iteration']}.npz" if choice["iteration"]!=20 else "weights.npz")
        shutil.copy2(weights,model/"word.npz")
    write_json(model/"model.json",{"schema_version":1,"variant":variant,"choice":choice,
        "parent":"previous_model","known_best_kaggle_user_report":.53985,"new_kaggle_score":None,
        "teacher_allows_pretrained_user_report":True,"weights_of_bert_updated":False,
        "history_table_scope":"query identities selected without labels; unseen identities fall back to round6" if variant=="history" else "not applicable",
        "word_training_rows":7531189 if variant=="sparse" else None})
    prefix="sparse_" if variant=="sparse" else ""
    frame=pd.read_csv(OUT/(prefix+"confirmation_predictions.csv"))
    epsilon=float(np.load(ROOT/"task2/outputs/round3_20261105/development.npz")["epsilon"])
    write_json(OUT/(prefix+"uncertainty.json"),uncertainty(frame,epsilon))
    print("模型包完成",model,flush=True)


def verify(variant):
    source,_,_=files(variant);reproduced=OUT/f"reproduced_{variant}.csv"
    first=pd.read_csv(source);second=pd.read_csv(reproduced)
    sample=pd.read_csv(ROOT/"comp-5434-20261-project-task-2/sampleSubmission.csv")
    assert list(first.columns)==["id","rating"] and len(first)==10000
    np.testing.assert_array_equal(first.id,sample.id)
    assert first.id.is_unique and np.isfinite(first.rating).all() and first.rating.between(1,5).all()
    np.testing.assert_allclose(first.rating,second.rating,atol=1e-8,rtol=0)
    old=json.loads((OUT/"old_submissions.json").read_text(encoding="utf-8"))
    assert all(sha256(ROOT/path)==digest for path,digest in old.items())
    output=ROOT/"submissions/20261114_task2_round7";output.mkdir(exist_ok=True)
    name="task2_posterior_history.csv" if variant=="history" else "task2_sparse_noise_fusion.csv"
    target=output/name
    if target.exists():raise FileExistsError(target)
    shutil.copy2(source,target)
    write_json(output/f"manifest_{variant}.json",{"rows":10000,"sha256":sha256(target),
        "old_submissions_unchanged":True,"old_files_checked":len(old),"independent_raw_csv_prediction":True,
        "independent_byte_identical":source.read_bytes()==reproduced.read_bytes(),
        "max_abs_prediction_difference":float(np.abs(first.rating-second.rating).max()),
        "new_kaggle_score":None,"previous_measured_best":.53985,"model":str((OUT/("model_"+variant)).relative_to(ROOT))})
    print("候选验证完成",target,"sha256",sha256(target),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--variant",choices=["history","sparse"],required=True)
    parser.add_argument("--stage",choices=["pack","verify"],required=True);args=parser.parse_args()
    (pack if args.stage=="pack" else verify)(args.variant)
