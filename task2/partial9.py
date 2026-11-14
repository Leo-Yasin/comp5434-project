"""第九轮：冻结DeBERTa前11层，训练最后一层与原五星分类头。

前缀不读取课程标签，也不建立反向图；缓存仅包含正文的token隐向量。
用相对位置嵌入的原LayerNorm参数恢复共享常量，不能把它误当成可训练新特征。
所有源参数来自第八轮已核验的固定公开版本，torch.load只允许张量。
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig,AutoModelForSequenceClassification
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Layer,ContextPooler,StableDropout,build_relative_position)
from .audit import ROOT,sha256,write_json

SOURCE=Path.home()/".cache/comp5434-english/model"
FOLDER=Path.home()/".cache/comp5434-local9"
OUT=ROOT/"task2/outputs/round9_20261114"


class TopLayer(nn.Module):
    """最后一层+原pooler/classifier；前缀隐向量是输入，原相对嵌入固定。"""
    def __init__(self,config,relative):
        super().__init__();self.config=config
        self.block=DebertaV2Layer(config);self.pooler=ContextPooler(config)
        self.classifier=nn.Linear(config.pooler_hidden_size,5)
        self.dropout=StableDropout(config.hidden_dropout_prob)
        self.register_buffer("relative",relative)

    def forward(self,hidden,mask):
        length=hidden.shape[1]
        # 同时屏蔽padding作为query/key；位置桶和原12层网络完全相同。
        attention=mask[:,None,None,:]*mask[:,None,:,None]
        position=build_relative_position(length,length,bucket_size=self.config.position_buckets,
            max_position=self.config.max_position_embeddings,device=hidden.device)
        result=self.block(hidden,attention,relative_pos=position,rel_embeddings=self.relative)
        logits=self.classifier(self.dropout(self.pooler(result)))
        return logits,result[:,0]


def source_model():
    """只删除已验证的旧版位置索引buffer，全部可学习参数严格加载。"""
    meta=json.loads((ROOT/"task2/english_model_source.json").read_text(encoding="utf-8"))
    expected=next(row["sha256"] for row in meta["files"] if row["file"]=="pytorch_model.bin")
    if sha256(SOURCE/"pytorch_model.bin")!=expected:raise ValueError("原始权重SHA不匹配。")
    config=AutoConfig.from_pretrained(SOURCE,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_config(config,trust_remote_code=False)
    state=torch.load(SOURCE/"pytorch_model.bin",map_location="cpu",weights_only=True)
    old=state.pop("deberta.embeddings.position_ids",None)
    if old is not None:torch.testing.assert_close(old,torch.arange(config.max_position_embeddings).expand(1,-1))
    model.load_state_dict(state,strict=True);model.eval();return model


def load_top(path=None,config_folder=None):
    saved=torch.load(path or FOLDER/"top_initial.pt",map_location="cpu",weights_only=True)
    config=AutoConfig.from_pretrained(config_folder or SOURCE,local_files_only=True,trust_remote_code=False)
    model=TopLayer(config,saved["relative"]);model.load_state_dict(saved,strict=True)
    return model


def frozen_prefix(device):
    """冻结完整前缀；只保留11个block，消除缓存时无用的最后层和分类器运算。"""
    model=source_model().deberta
    model.encoder.layer=nn.ModuleList(list(model.encoder.layer)[:11])
    model.requires_grad_(False);model.eval();return model.to(device)


def prepare():
    """拆出最后层，并从原FP32 ONNX反向遍历依赖，导出只含前11层的图。"""
    import onnx
    FOLDER.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (FOLDER/"top_initial.pt").exists():raise FileExistsError("已拆分，禁止覆盖初始权重。")
    model=source_model()
    with torch.no_grad():relative=model.deberta.encoder.get_rel_embedding().clone()
    top=TopLayer(model.config,relative)
    top.block.load_state_dict(model.deberta.encoder.layer[11].state_dict(),strict=True)
    top.pooler.load_state_dict(model.pooler.state_dict(),strict=True)
    top.classifier.load_state_dict(model.classifier.state_dict(),strict=True)
    torch.save(top.state_dict(),FOLDER/"top_initial.pt")
    model.config.save_pretrained(FOLDER/"config")
    del model,top;gc.collect()
    graph=onnx.load(SOURCE/"model.onnx")
    tensor="/deberta/encoder/layer.10/output/LayerNorm/LayerNormalization_output_0"
    if sum(tensor in node.output for node in graph.graph.node)!=1:raise ValueError("前缀边界不唯一。")
    needed={tensor};nodes=[]
    for node in reversed(graph.graph.node):
        if any(name in needed for name in node.output):nodes.append(node);needed.update(node.input)
    kept=[v for v in graph.graph.initializer if v.name in needed]
    graph.graph.ClearField("node");graph.graph.node.extend(reversed(nodes))
    graph.graph.ClearField("initializer");graph.graph.initializer.extend(kept)
    graph.graph.ClearField("output")
    graph.graph.output.extend([onnx.helper.make_tensor_value_info(tensor,onnx.TensorProto.FLOAT,["batch","sequence",768])])
    onnx.checker.check_model(graph);onnx.save(graph,FOLDER/"prefix.onnx")
    write_json(OUT/"source.json",{"source_revision":"06f020e0dbf909570eb886423bd3af256b855546",
        "prefix_sha256":sha256(FOLDER/"prefix.onnx"),"top_sha256":sha256(FOLDER/"top_initial.pt"),
        "frozen_encoder_layers":11,"trainable_encoder_layers":1,"teacher_permits_pretrained":True,
        "prefix_output":tensor,"initial_source_not_updated":True})
    print("第九轮前缀和最后层已拆分",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["prepare"],required=True)
    args=p.parse_args();torch.set_num_threads(1);prepare()
