"""借鉴SmoothQuant的等价逐通道缩放，为DeBERTa做保守的CPU量化。

方法来源：Xiao等，SmoothQuant，arXiv:2211.10438。这里自行实现ONNX变换，
不是复制官方实现，也不声称论文已证明本数据上有效。全部校准只读内部训练正文。
"""
import gc
import json
import time
import numpy as np
import pandas as pd
from .audit import ROOT,write_json,sha256
from .english_prepare import FOLDER


def channel_scale(weight,activation,alpha=.5):
    """正尺度仅由无标签激活和权重幅值决定；轴0是MatMul输入通道。"""
    weight=np.asarray(weight);activation=np.asarray(activation)
    if weight.ndim!=2 or activation.shape!=(weight.shape[0],) or not 0<=alpha<=1:
        raise ValueError("缩放轴/alpha不合法。")
    if not np.isfinite(weight).all() or not np.isfinite(activation).all() or np.any(activation<0):
        raise ValueError("激活边界必须有限且非负。")
    return (np.maximum(activation,1e-5)**alpha/np.maximum(np.abs(weight).max(axis=1),1e-5)**(1-alpha)).clip(1e-3,1e3).astype(np.float32)


def smooth_graph(graph,maxima):
    """为每个被缩放分支建立独立权重，原常量及其Identity别名始终保留原值。

    DeBERTa的内容投影和相对位置投影可能通过Identity共享同一常量。
    直接原位改该常量会破坏未同步缩放的分支，因此不能以“直接消费者”推断独占。
    """
    from onnx import numpy_helper,helper
    initializers={t.name:t for t in graph.graph.initializer};nodes=[];count=0
    for number,node in enumerate(graph.graph.node):
        if node.op_type=="MatMul" and node.input[1] in initializers:
            name=node.name.strip("/").rsplit("/",1)[0].replace("/",".")
            if name not in maxima:raise ValueError("未能对应Linear校准模块："+name)
            weight=numpy_helper.to_array(initializers[node.input[1]])
            scale=channel_scale(weight,maxima[name]);count+=1
            scale_name=f"smooth_scale_{number}";weight_name=f"smooth_weight_{number}";output=f"smooth_input_{number}"
            graph.graph.initializer.append(numpy_helper.from_array(1./scale,name=scale_name))
            graph.graph.initializer.append(numpy_helper.from_array((weight*scale[:,None]).astype(np.float32),name=weight_name))
            nodes.append(helper.make_node("Mul",[node.input[0],scale_name],[output],name=f"smooth_mul_{number}"))
            node.input[0]=output;node.input[1]=weight_name
        nodes.append(node)
    del graph.graph.node[:];graph.graph.node.extend(nodes)
    used={name for node in graph.graph.node for name in node.input}|{x.name for x in graph.graph.output}
    kept=[item for item in graph.graph.initializer if item.name in used]
    del graph.graph.initializer[:];graph.graph.initializer.extend(kept)
    return graph,count


def collect():
    """记录每个Linear输入通道的绝对最大值；不使用课程标签或反向传播。"""
    import torch
    from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
    torch.set_num_threads(3);torch.set_num_interop_threads(1)
    data=np.load(ROOT/"task2/outputs/round3_20261105/development.npz")
    index=np.load(ROOT/"task2/outputs/round6_20261108/selection.npz")["index"]
    groups=data["groups"][index]
    fit=(groups%np.uint64(100)==0)&((groups//np.uint64(100))%np.uint64(5)!=0)
    frame=pd.read_csv(ROOT/"task2/outputs/round6_20261108/development_text.csv",keep_default_na=False)
    texts=sorted(frame.loc[fit,"comment"].iloc[:128].astype(str).tolist(),key=len)
    tokenizer=AutoTokenizer.from_pretrained(FOLDER/"tokenizer_spm",local_files_only=True,use_fast=False,trust_remote_code=False)
    config=AutoConfig.from_pretrained(FOLDER,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_config(config,trust_remote_code=False)
    state=torch.load(FOLDER/"pytorch_model.bin",map_location="cpu",weights_only=True)
    ids=state.pop("deberta.embeddings.position_ids",None)
    if ids is not None:torch.testing.assert_close(ids,torch.arange(config.max_position_embeddings).expand(1,-1))
    model.load_state_dict(state,strict=True);del state;gc.collect();model.eval()
    maxima={};handles=[]
    def hook(name):
        def record(module,inputs,outputs):
            value=inputs[0].detach().reshape(-1,inputs[0].shape[-1]).abs().amax(dim=0).numpy()
            maxima[name]=np.maximum(maxima[name],value) if name in maxima else value.copy()
        return record
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Linear):handles.append(module.register_forward_hook(hook(name)))
    started=time.perf_counter()
    for first in range(0,len(texts),8):
        encoded=tokenizer(texts[first:first+8],padding=True,truncation=True,max_length=128,return_tensors="pt")
        with torch.no_grad():model(**encoded)
    for handle in handles:handle.remove()
    np.savez_compressed(FOLDER/"activation_maxima.npz",**maxima)
    write_json(FOLDER/"activation_calibration.json",{"rows":128,"labels_used":False,"input_scope":"inner_fit_only",
        "seconds":time.perf_counter()-started,"linear_modules":len(maxima)})
    print("量化激活校准完成",len(maxima),time.perf_counter()-started,flush=True)


def convert():
    """对每个常量MatMul权重B做B'=diag(s)B，并在输入插入x'=x/diag(s)。

    所以x'B'=xB（浮点舍入误差除外）。s=amax^0.5/wmax^0.5，平衡激活/权重离群值。
    不量化Gemm：保留分类头、池化器和相对位置投影的FP32计算。
    """
    import onnx
    from onnxruntime.quantization import quantize_dynamic,QuantType
    maxima=dict(np.load(FOLDER/"activation_maxima.npz"));started=time.perf_counter()
    graph,count=smooth_graph(onnx.load(str(FOLDER/"model.onnx")),maxima)
    target=FOLDER/"model_smooth_v2.onnx";onnx.save(graph,str(target));del graph;gc.collect()
    import onnxruntime as ort
    reference=np.load(FOLDER/"export_reference.npz");feed={k:reference[k] for k in ["input_ids","attention_mask"]}
    options=ort.SessionOptions();options.intra_op_num_threads=2
    check=ort.InferenceSession(str(target),sess_options=options,providers=["CPUExecutionProvider"])
    actual,hidden=check.run(None,feed)
    np.testing.assert_allclose(actual,reference["logits"],atol=3e-4,rtol=3e-4)
    np.testing.assert_allclose(hidden,reference["cls_embedding"],atol=5e-4,rtol=5e-4)
    fp_error=float(np.abs(actual-reference["logits"]).max());del check;gc.collect()
    quantized=FOLDER/"model_smooth_v2_int8.onnx"
    quantize_dynamic(str(target),str(quantized),weight_type=QuantType.QInt8,per_channel=True,reduce_range=True,
        op_types_to_quantize=["MatMul"],extra_options={"MatMulConstBOnly":True})
    write_json(FOLDER/"smoothing_v2.json",{"alpha":.5,"linear_weights_scaled":count,"labels_used":False,
        "seconds":time.perf_counter()-started,"source":"https://arxiv.org/abs/2211.10438",
        "implementation":"own ONNX explicit Mul and matching row-scaled weight; Gemm kept FP32",
        "fp32_function_preserved_verified":True,"fp32_max_logit_error":fp_error,"quantized_sha256":sha256(quantized)})
    write_json(FOLDER/"quantization_choice.json",{"selected":{"file":quantized.name},"sha256":sha256(quantized),
        "needs_broad_audit_before_use":True,"selection_used_no_course_labels":True})
    print("平滑量化网络已生成，尚待审计",time.perf_counter()-started,flush=True)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["collect","convert"],required=True);args=p.parse_args()
    globals()[args.stage]()
