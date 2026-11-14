"""一次联合学习的第八轮对照：全部输入在原外层验证之外拟合或完全冻结。

不把已在当前内部训练行拟合的旧预测头logits作为训练偏置，避免层层残差
只看到父模型的训练内误差。旧第七轮只作为验证/测试的可选评分融合对象。
"""
import numpy as np
from .noise_softmax import softmax,predict_channel
from .semantic_fusion import semantic_features
from .embedding_heads import fit_mlp,mlp_predict,fit_linear


def frozen_prior(bert_logits,english_logits=None):
    """预先固定等权混合两个冻结类别分布，和课程评分无关。"""
    probability=softmax(np.asarray(bert_logits,dtype=np.float64))
    if english_logits is not None:probability=.5*probability+.5*softmax(np.asarray(english_logits,dtype=np.float64))
    return np.log(np.maximum(probability,1e-15))


def assemble(structure,auxiliary,bert_logits,bert_cls,lexical_logits,english_logits,english_cls):
    """保留原35列独立结构特征、辅助词评分和两种CLS，不包含旧监督残差树评分。"""
    n=len(structure);neutral=np.full(n,3.)
    blocks=[("structure",np.column_stack([structure,auxiliary])),("bert_cls",bert_cls),
        ("bert_distribution",semantic_features(bert_logits,neutral)),
        ("lexical",semantic_features(lexical_logits,neutral)),
        ("english_distribution",semantic_features(english_logits,neutral)),("english_cls",english_cls)]
    layout={};at=0
    for name,block in blocks:
        if len(block)!=n:raise ValueError("联合输入行数错位。")
        layout[name]=[at,at+block.shape[1]];at+=block.shape[1]
    features=np.column_stack([value for _,value in blocks]).astype(np.float32)
    if not np.isfinite(features).all():raise ValueError("联合输入非有限。")
    return features,frozen_prior(bert_logits,english_logits),layout


def keep_columns(layout,remove):
    """按具名分块移除输入，避免消融时切错768维块或仍保留对应偏置。"""
    return np.concatenate([np.arange(first,last) for name,(first,last) in layout.items() if name not in remove])


def fit(features,offset,rating,mask,epsilon,choice):
    """沿用已有限差分验证的自主噪声损失与MLP反向传播。"""
    # 分块累计标准化矩，避免最终83047×1629特征额外常驻一份完整float64副本。
    anchor,report=fit_linear(features,offset,rating,mask,epsilon,ridge=choice["ridge"],iterations=250)
    if choice["family"]=="joint_linear":return anchor,report
    states,mlp_report=fit_mlp(features,offset,rating,mask,epsilon,anchor,epochs=choice["epochs"])
    return states[choice["epochs"]],{"anchor":report,"mlp":mlp_report}


def clean_prediction(features,offset,state,choice):
    return predict_channel(features,offset,state) if choice["family"]=="joint_linear" else mlp_predict(features,offset,state)
