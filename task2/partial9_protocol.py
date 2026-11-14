"""不依赖PyTorch的第九轮路径、固定分组和预登记候选范围。"""
from pathlib import Path
import numpy as np
from .audit import ROOT

FOLDER=Path.home()/".cache/comp5434-local9"
OUT=ROOT/"task2/outputs/round9_20261114"


def tag(rate):return f"lr{rate:.0e}"


def fit_mask(groups,stage):
    """分组只由原无标签哈希决定；确认阶段不包含group1标签。"""
    group0=groups%100==0;check=group0&((groups//100)%5==0)
    if stage=="inner":return group0&~check
    if stage=="confirm":return group0
    if stage=="final":return np.ones(len(groups),bool)
    raise ValueError("未知阶段。")


def teacher_posterior(probability,observed_index,epsilon):
    """Bayes去噪后验：固定旧模型q作为潜在标签先验，观测y通过均匀替换通道产生。"""
    q=np.asarray(probability,dtype=np.float64);y=np.asarray(observed_index,dtype=np.int64)
    if q.shape!=(len(y),5) or not np.isfinite(q).all() or (q<0).any() or not np.isin(y,np.arange(5)).all():
        raise ValueError("后验输入不合法。")
    if not np.allclose(q.sum(1),1,atol=1e-6) or not 0<=epsilon<1:raise ValueError("先验/噪声率不合法。")
    likelihood=np.full_like(q,epsilon/5);likelihood[np.arange(len(y)),y]+=1-epsilon
    posterior=q*likelihood;total=posterior.sum(1,keepdims=True)
    if (total<=0).any():raise ValueError("观测在该先验/通道下概率为零。")
    posterior/=total
    return posterior.astype(np.float32)


def objective_suffix(objective):
    return {"forward_noise":"","teacher_posterior":"_posterior","residual_noise":"_residual"}[objective]


def run_name(stage,choice,classifier_only=False):
    return f"{stage}_{tag(choice['learning_rate'])}"+objective_suffix(choice.get("objective","forward_noise"))+("_classifier_only" if classifier_only else "")


PROTOCOL={"learning_rates":[1e-5,3e-5],"snapshots":[1,2],"planned_epochs":2,"inference_max_lengths":[128,256],
    "direct_alphas":[.25,.5,.75,1.],"noise_head_ridges":[100.,1000.],"adapted_cls_ridges":[1000.,5000.],"candidate_count":128,
    "objective_runs":[{"objective":"forward_noise","learning_rate":1e-5},{"objective":"forward_noise","learning_rate":3e-5},
                      {"objective":"teacher_posterior","learning_rate":1e-5},{"objective":"residual_noise","learning_rate":3e-5}],
    "effective_batch":64,"micro_batch":16,"weight_decay":.01,"classifier_lr_multiplier":5,
    "seed":5434,"prefix_frozen_layers":11,"trainable_encoder_layers":1,"known_best_kaggle":.50646,
    "historical_validation_reuse":True,"confirmation_used_for_selection":False,
    "long_context_added_before_first_new_neural_scores":True,"long_context_diagnostic_used_internal_baseline_only":True,
    "posterior_added_before_first_new_validation_metric":True,"teacher_training_predictions_not_cross_fitted_within_fit":True,
    "adapted_cls_added_after_first_snapshot_fixed_blend_diagnostic":True,
    "residual_added_after_first_snapshot_diagnostic_before_confirmation":True}
