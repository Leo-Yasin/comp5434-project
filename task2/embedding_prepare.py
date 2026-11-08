"""第六轮准备：公开冻结BERT的768维CLS表示，按历史支持条件选取输入。

只给原ONNX图添加输出节点，不修改任意预训练权重。仅用无标签历史支持标记
挑选与测试条件相同的记录；标签只保存在原元数据，不写进网络推理输入。
运行在Python3.12语义环境，需要onnx==1.19.1。
"""
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import onnx
from onnx import helper, TensorProto

from .audit import ROOT, write_json, sha256

OUT = ROOT/"task2/outputs/round6_20261108"
OLD = ROOT/"task2/outputs/round5_20261105"


def main():
    OUT.mkdir(exist_ok=False)
    source = OLD/"external_model/onnx/model_quantized.onnx"
    graph = onnx.load(str(source))
    tensor = "/bert/pooler/Gather_output_0"
    if tensor not in {name for node in graph.graph.node for name in node.output}:
        raise ValueError("固定导出版本的CLS节点不匹配，不能猜测中间层。")
    # Gather从最后一层隐藏状态取位置0，即[CLS]；不是平均池化或原五维分类结果。
    graph.graph.node.append(helper.make_node("Identity", [tensor], ["cls_embedding"], name="expose_cls_embedding"))
    graph.graph.output.append(helper.make_tensor_value_info("cls_embedding", TensorProto.FLOAT, ["batch_size",768]))
    onnx.checker.check_model(graph)
    model = OUT/"encoder.onnx"
    onnx.save(graph, str(model))
    original = np.load(ROOT/"task2/outputs/round3_20261105/development.npz")
    context = original["context"]
    # 仅取三种身份都已有训练历史的样本；保留原group号0/1边界和内部80/20边界。
    selected = np.flatnonzero(np.all(context[:,[2,5,8]] == 0, axis=1))
    texts = pd.read_csv(OLD/"validation_text.csv", keep_default_na=False)
    if not np.array_equal(texts.id.to_numpy(), original["id"]):
        raise ValueError("验证文本与固定元数据错位。")
    texts.iloc[selected].to_csv(OUT/"development_text.csv", index=False)
    # 与旧推理相同的CSV字节格式，保证测试语义logits的受控比较可核对。
    (OUT/"test_text.csv").write_bytes((OLD/"test_text.csv").read_bytes())
    np.savez_compressed(OUT/"selection.npz", index=selected, id=original["id"][selected])
    g = original["groups"][selected]
    dev = g%np.uint64(100)==0
    inner_check=dev&((g//np.uint64(100))%np.uint64(5)==0)
    # 先登记有限候选，确认结果不得再用于改层数、噪声率、轮数或融合幅度。
    write_json(OUT/"protocol.json", {
        "registered_at": datetime.datetime.now().astimezone().isoformat(),
        "date_user_facing":"2026-11-08", "known_best_kaggle_user_report":.54604,
        "external_model_allowed_teacher_confirmation_user_report":True,
        "condition":"must understand the pretrained model", "weights_changed":False,
        "source_onnx_sha256":sha256(source), "encoder_sha256":sha256(model),
        "source_revision":"fad9b372f770c208d2efaf00d02f2cb3b61ec450",
        "feature":"last hidden state at CLS, before pooler Dense and Tanh", "dimension":768,
        "rows":len(selected), "inner_fit":int((dev&~inner_check).sum()),
        "inner_check":int(inner_check.sum()), "confirmation":int((~dev).sum()),
        "sampling_uses_labels":False, "historical_validation_reuse_disclosed":True,
        "candidates":{"ridge_residual":[100.,1000.],"noise_linear":[100.,1000.],
            "noise_mlp":{"linear_anchor_ridge":1000.,"hidden":64,"dropout":.2,
                "learning_rate":.001,"l2":.001,"epochs":[3,6,12]}},
        "mixing_alpha":[.5,1.], "primary_selection_metric":"uniform_noise_adjusted_risk",
        "note":"all selected records match test known-identity condition; no title-proxy selection"})
    print("第六轮准备完成", json.loads((OUT/"protocol.json").read_text(encoding="utf-8")), flush=True)


if __name__=="__main__":
    main()
