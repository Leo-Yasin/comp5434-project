"""只读解释最终MLP中的线性锚点贡献，不改变已经冻结的候选。

锚点参数正是在confirm阶段MLP训练前保存的线性通道，移除MLP残差即可得到此对照。
本脚本没有fit操作，不用结果重新选择轮数或超参数。
"""
import numpy as np
from .audit import write_json
from .embedding_features import OUT,development,stage_features
from .embedding_heads import linear_logits
from .noise_softmax import softmax
from .round3_train import evaluate


def main():
    data,index,logits,embedding=development()
    features,parent=stage_features(index,logits,embedding,"confirm")
    saved=dict(np.load(OUT/"confirm_head.npz"))
    anchor={key:saved["anchor_"+key] for key in ["center","scale","weights"]}
    prediction=softmax(linear_logits(features,logits,anchor))@np.arange(1.,6.)
    confirm=data["groups"]%np.uint64(100)==1
    metrics=evaluate(prediction[confirm],data["y"][confirm],data["stars"][confirm],float(data["epsilon"]))
    write_json(OUT/"linear_anchor_confirmation_read_only.json",{"not_used_to_reselect":True,"metrics":metrics})
    print(metrics,flush=True)


if __name__=="__main__":main()
