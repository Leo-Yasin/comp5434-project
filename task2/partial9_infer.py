"""自包含推理：CPU冻结前缀+FP16缓存+FP32最后层，输出原始logits及可选CLS。

残差目标的原始输出还需由round9_predict结合第八轮类别偏置，不能单独当评分。
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .partial9 import load_top
from .partial9_cache import run
from .partial9_train import predict
from .audit import sha256,write_json


def main():
    p=argparse.ArgumentParser()
    for name in ["model","input","cache-folder","output"]:p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);began=time.perf_counter()
    runtime=json.loads((args.model/"runtime.json").read_text(encoding="utf-8"))
    expected={"prefix_backend":"openvino","prefix_batch":16,"hidden_dtype":"float16","top_backend":"torch_cpu","top_batch":16}
    if any(runtime.get(k)!=v for k,v in expected.items()) or runtime.get("max_length") not in [128,256]:
        raise ValueError("不支持的适配推理配置。")
    top_threads=int(runtime.get("top_threads",1))
    if top_threads not in [1,4]:raise ValueError("未核验的CPU最后层线程数。")
    # 前缀缓存验证输入、图、分词器及运行时指纹；只复用冻结计算，不复用最终评分。
    previous=0
    if (args.cache_folder/"state.json").exists():previous=json.loads((args.cache_folder/"state.json").read_text(encoding="utf-8"))["completed"]
    run(args.input,args.cache_folder,"openvino",16,prefix_path=args.model/"prefix.onnx",tokenizer_folder=args.model/"tokenizer",max_length=runtime["max_length"])
    # 前缀与最后层顺序执行；前缀结束后才启用经逐位一致性审计的CPU线程数。
    torch.set_num_threads(top_threads)
    model=load_top(args.model/"top.pt",args.model/"config").eval()
    if runtime.get("export_cls",False):
        from .partial9_cls import extract
        values,embedding=extract(model,args.cache_folder,device="cpu")
        np.save(args.output.with_suffix(".cls.npy"),embedding)
    else:values=predict(model,args.cache_folder,batch_size=16,device="cpu")
    np.savez_compressed(args.output,id=np.load(args.cache_folder/"id.npy"),logits=values)
    write_json(args.output.with_suffix(".timing.json"),{"seconds":time.perf_counter()-began,"rows":len(values),
        "prefix_rows_resumed":previous,"prefix_sha256":sha256(args.model/"prefix.onnx"),"top_sha256":sha256(args.model/"top.pt"),
        "prefix_backend":"openvino_f32","hidden_dtype":"float16","top_backend":"torch_cpu_f32","top_threads":top_threads})


if __name__=="__main__":main()
