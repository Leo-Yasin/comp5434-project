"""第八轮英语评论模型：固定公开版本下载、本地CPU导出和INT8校验。

仅下载模型权重和分词配置，不下载任何原始评论或源评分答案。
新的大文件均保存在C盘用户缓存，避免挤占项目F盘。
"""
import argparse
import gc
import json
from pathlib import Path
import time
from .audit import ROOT,sha256,write_json

REPO="patrickvonplaten/deberta_amazon_reviews_v1"
REVISION="06f020e0dbf909570eb886423bd3af256b855546"
FOLDER=Path.home()/".cache/comp5434-english/model"


def download(proxy=None):
    import requests
    FOLDER.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    if proxy:session.proxies={"https":proxy}
    source=json.loads((ROOT/"task2/english_model_source.json").read_text(encoding="utf-8"))
    expected={record["file"]:record for record in source["files"]}
    if source["repository"]!=REPO or source["revision"]!=REVISION:raise ValueError("源清单版本不一致。")
    names=["README.md","config.json","tokenizer_config.json","special_tokens_map.json","added_tokens.json","spm.model","pytorch_model.bin"]
    records=[]
    for name in names:
        target=FOLDER/name;started=time.perf_counter()
        if not target.exists():
            temporary=target.with_suffix(target.suffix+".partial")
            with session.get(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}",timeout=(30,90),stream=True) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    size=0;last=0
                    for block in response.iter_content(4*1024*1024):
                        if block:stream.write(block);size+=len(block)
                        if size-last>=64*1024*1024:print("英语模型下载",name,size,flush=True);last=size
            temporary.replace(target)
        records.append({"file":name,"bytes":target.stat().st_size,"sha256":sha256(target),"seconds":time.perf_counter()-started})
        if target.stat().st_size!=expected[name]["bytes"] or records[-1]["sha256"]!=expected[name]["sha256"]:
            raise ValueError("已存在/下载的模型文件不匹配固定版本："+name)
        write_json(FOLDER/"provenance.json",{"repository":REPO,"revision":REVISION,"license":"MIT",
            "source_card_incomplete":True,"related_author_tutorial":"https://huggingface.co/blog/supercharge-customer-service-with-machine-learning",
            "training_corpus_overlap":"unknown","teacher_allows_pretrained_user_report":True,"files":records})
        print("英语模型文件就绪",name,flush=True)


def export():
    import torch
    from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
    import numpy as np
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    target=FOLDER/"model.onnx"
    if target.exists():raise FileExistsError(target)
    started=time.perf_counter()
    config=AutoConfig.from_pretrained(FOLDER,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_config(config,trust_remote_code=False)
    # 显式只加载张量权重；不执行仓库自定义代码或training_args等pickle对象。
    weights=torch.load(FOLDER/"pytorch_model.bin",map_location="cpu",weights_only=True)
    # 旧版Transformers曾把固定位置索引作为持久buffer保存，新版改为非持久buffer。
    # 只允许删除这一个经逐值核验的派生常量，其他缺失/多余参数仍严格报错。
    position=weights.pop("deberta.embeddings.position_ids",None)
    if position is not None:
        torch.testing.assert_close(position,torch.arange(config.max_position_embeddings).expand(1,-1))
    model.load_state_dict(weights,strict=True);del weights;gc.collect();model.eval()
    tokenizer=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True,use_fast=False,trust_remote_code=False)
    texts=["This is excellent. It works perfectly.","This is terrible and does not work."]
    encoded=tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors="pt")
    class Wrapper(torch.nn.Module):
        def __init__(self,network):super().__init__();self.network=network
        def forward(self,input_ids,attention_mask):
            hidden=self.network.deberta(input_ids=input_ids,attention_mask=attention_mask)[0]
            pooled=self.network.pooler(hidden)
            logits=self.network.classifier(self.network.dropout(pooled))
            return logits,hidden[:,0]
    wrapper=Wrapper(model)
    with torch.no_grad():
        result=wrapper(encoded["input_ids"],encoded["attention_mask"])
        expected=result[0].numpy();expected_cls=result[1].numpy()
        np.testing.assert_allclose(expected,model(**encoded).logits.numpy(),atol=1e-6,rtol=1e-6)
    torch.onnx.export(wrapper,(encoded["input_ids"],encoded["attention_mask"]),str(target),opset_version=17,
        input_names=["input_ids","attention_mask"],output_names=["logits","cls_embedding"],
        dynamic_axes={"input_ids":{0:"batch",1:"sequence"},"attention_mask":{0:"batch",1:"sequence"},"logits":{0:"batch"},"cls_embedding":{0:"batch"}},
        do_constant_folding=True,dynamo=False)
    tokenizer.save_pretrained(FOLDER/"tokenizer_spm")
    np.savez_compressed(FOLDER/"export_reference.npz",**{k:v.numpy() for k,v in encoded.items() if k in ["input_ids","attention_mask"]},logits=expected,cls_embedding=expected_cls)
    del model,wrapper;gc.collect()
    write_json(FOLDER/"export.json",{"seconds":time.perf_counter()-started,"onnx_sha256":sha256(target),"torch":torch.__version__,
        "weights_updated":False,"inputs":["input_ids","attention_mask"],"source_revision":REVISION})
    print("英语模型CPU导出完成",time.perf_counter()-started,flush=True)


def quantize():
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic,QuantType
    target=FOLDER/"model_int8.onnx"
    if target.exists():raise FileExistsError(target)
    started=time.perf_counter()
    # 原预训练使用SentencePiece byte fallback，保留慢分词器，避免Fast转换丢失罕见字符。
    import torch
    from transformers import AutoTokenizer,AutoConfig,AutoModelForSequenceClassification
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(FOLDER,use_fast=False,local_files_only=True,trust_remote_code=False)
    tokenizer.save_pretrained(FOLDER/"tokenizer_spm")
    config=AutoConfig.from_pretrained(FOLDER,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_config(config,trust_remote_code=False)
    weights=torch.load(FOLDER/"pytorch_model.bin",map_location="cpu",weights_only=True)
    position=weights.pop("deberta.embeddings.position_ids",None)
    if position is not None:torch.testing.assert_close(position,torch.arange(config.max_position_embeddings).expand(1,-1))
    model.load_state_dict(weights,strict=True);del weights;gc.collect();model.eval()
    references=[]
    for number,texts,length in [(0,["Good item."],8),(1,["Works well.","Broken and useless.","It is average."],40),
                                (2,["Excellent product. "*80]*8,128)]:
        encoded=tokenizer(texts,padding="max_length",truncation=True,max_length=length,return_tensors="pt")
        with torch.no_grad():
            hidden=model.deberta(input_ids=encoded["input_ids"],attention_mask=encoded["attention_mask"])[0]
            logits=model.classifier(model.dropout(model.pooler(hidden)))
        refs={k:encoded[k].numpy() for k in ["input_ids","attention_mask"]}
        references.append((refs,logits.numpy(),hidden[:,0].numpy()))
    del model;gc.collect()
    quantize_dynamic(str(FOLDER/"model.onnx"),str(target),weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul","Gemm"],per_channel=False)
    reference=np.load(FOLDER/"export_reference.npz");feed={k:reference[k] for k in ["input_ids","attention_mask"]}
    options=ort.SessionOptions();options.intra_op_num_threads=1
    original=ort.InferenceSession(str(FOLDER/"model.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
    fp,fp_cls=original.run(None,feed)
    dynamic_errors=[]
    for sample,expected,expected_cls in references:
        actual,actual_cls=original.run(None,sample)
        np.testing.assert_allclose(actual,expected,atol=3e-4,rtol=3e-4)
        np.testing.assert_allclose(actual_cls,expected_cls,atol=5e-4,rtol=5e-4)
        dynamic_errors.append({"shape":list(sample["input_ids"].shape),"max_logit_error":float(np.abs(actual-expected).max())})
    del original;gc.collect()
    quantized=ort.InferenceSession(str(target),sess_options=options,providers=["CPUExecutionProvider"])
    actual,actual_cls=quantized.run(None,feed)
    np.testing.assert_allclose(fp,reference["logits"],atol=2e-4,rtol=2e-4)
    np.testing.assert_allclose(fp_cls,reference["cls_embedding"],atol=3e-4,rtol=3e-4)
    if not np.isfinite(actual).all():raise ValueError("量化结果非有限。")
    write_json(FOLDER/"quantization.json",{"seconds":time.perf_counter()-started,"onnx_sha256":sha256(target),
        "onnxruntime":ort.__version__,"float_export_max_abs_error":float(np.abs(fp-reference["logits"]).max()),
        "pilot_quantization_max_abs_error":float(np.abs(actual-fp).max()),"pilot_cls_max_abs_error":float(np.abs(actual_cls-fp_cls).max()),
        "pilot_predicted_classes":actual.argmax(axis=1).tolist(),"dynamic_shape_checks":dynamic_errors,
        "quantization_not_guaranteed_lossless":True})
    print("英语INT8量化就绪",time.perf_counter()-started,flush=True)


def repair():
    """检测到默认INT8精度崩坏后，仅以原FP32网络输出为参照修正量化。

    与比赛标签无关，不能用确认组来挑量化方式。U8S8在部分CPU存在饱和问题，
    依次比较缩减范围/逐通道和U8U8，只发布通过语义一致性门槛的网络。
    """
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic,QuantType
    reference=np.load(FOLDER/"export_reference.npz");feed={k:reference[k] for k in ["input_ids","attention_mask"]}
    records=[]
    for label,kind,per_channel,reduce_range in [("reduced",QuantType.QInt8,True,True),("uint8",QuantType.QUInt8,False,False)]:
        destination=FOLDER/f"model_{label}.onnx";started=time.perf_counter()
        if not destination.exists():
            quantize_dynamic(str(FOLDER/"model.onnx"),str(destination),weight_type=kind,
                per_channel=per_channel,reduce_range=reduce_range,op_types_to_quantize=["MatMul","Gemm"])
        options=ort.SessionOptions();options.intra_op_num_threads=1
        model=ort.InferenceSession(str(destination),sess_options=options,providers=["CPUExecutionProvider"])
        logits,hidden=model.run(None,feed)
        record={"file":destination.name,"seconds":time.perf_counter()-started,
            "logit_error":float(np.abs(logits-reference["logits"]).max()),
            "cls_error":float(np.abs(hidden-reference["cls_embedding"]).max()),
            "classes_match":bool(np.array_equal(logits.argmax(axis=1),reference["logits"].argmax(axis=1)))}
        records.append(record);print("英语量化修复对照",record,flush=True)
        del model;gc.collect()
        if record["classes_match"] and record["logit_error"]<.5 and record["cls_error"]<1.5:
            write_json(FOLDER/"quantization_choice.json",{"selected":record,"trials":records,
                "sha256":sha256(destination),"selection_used_no_course_labels":True})
            return
    write_json(FOLDER/"quantization_rejected.json",{"trials":records,"fp32_fallback_required":True})
    raise RuntimeError("量化精度未通过，必须使用FP32或停止英语路线，不能当成同等模型。")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["download","export","quantize","repair"],required=True)
    parser.add_argument("--proxy",help="需要时显式指定本机代理；不把维护者的7897端口强加给队友。")
    args=parser.parse_args()
    if args.stage=="download":download(args.proxy)
    else:globals()[args.stage]()
