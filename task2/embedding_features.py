"""第六轮特征：重建同输入条件下第五轮基线，再拼接冻结CLS表示。

验证用相应inner/confirm旧模型，不能拿已经见过确认标签的final树作确认基线。
原图增加输出不改变logits；由于本轮只推理有历史子集，量化批组成可能与上轮不同，
本轮基线和新头使用相同logits与输入，避免把批量量化变化误当成新头收益。
"""
import json
import numpy as np
from .audit import ROOT
from .semantic_fusion import semantic_features, predict_linear
from .regression_boost import ResidualHistogramBoost

OUT=ROOT/"task2/outputs/round6_20261108"
OLD=ROOT/"task2/outputs/round5_20261105"


def combine(previous_features,previous_prediction,logits,embedding,linear,booster,alpha=1.):
    sf=semantic_features(logits,previous_prediction)
    x=np.column_stack([previous_features,previous_prediction,sf]).astype(np.float32)
    initial=predict_linear(previous_prediction,sf,linear)
    changed=booster.predict(x,initial,400)
    parent=np.clip(initial+alpha*(changed-initial),1,5)
    features=np.column_stack([x,parent,np.asarray(embedding)]).astype(np.float32)
    if features.shape[1]!=828 or not np.isfinite(features).all():raise ValueError("第六轮特征维度或数值错误。")
    return features,parent


def development():
    original=dict(np.load(ROOT/"task2/outputs/round3_20261105/development.npz"))
    index=np.load(OUT/"selection.npz")["index"];n=len(original["id"])
    data={k:v[index] if v.ndim and v.shape[0]==n else v for k,v in original.items()}
    semantic=np.load(OUT/"development.npz")
    if not np.array_equal(data["id"],semantic["id"]):raise ValueError("第六轮验证CLS的id错位。")
    embedding=np.load(OUT/"development.embedding.npy",mmap_mode="r")
    return data,index,semantic["logits"],embedding


def stage_features(index,logits,embedding,stage):
    cached=np.load(OLD/f"prior_{stage}.npz")
    if index is None:
        previous_features=cached["test_features"];previous_prediction=cached["test_prediction"]
    else:
        previous_features=cached["features"][index];previous_prediction=cached["prediction"][index]
    linear=dict(np.load(OLD/f"{stage}_linear.npz"))
    booster=ResidualHistogramBoost.from_dict(json.loads((OLD/f"{stage}_booster.json").read_text(encoding="utf-8")))
    return combine(previous_features,previous_prediction,logits,embedding,linear,booster)
