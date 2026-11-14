"""核验CUDA预测批量64与原批量16的数值一致性，完全不读取检查/确认评分。

只改变无梯度的128token前向；256token、训练微批和CPU部署仍沿用16。
实际适配快照的logits/CLS误差通过后才写启用记录，失败不启用大批量。
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import time
import numpy as np
import torch
from task2.partial9 import load_top
from task2.partial9_protocol import FOLDER,OUT
from task2.partial9_cls import extract
from task2.audit import sha256,write_json


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    controls=np.load(OUT/"controls.npz");folder=FOLDER/"development"
    fit=np.flatnonzero((controls["groups"]%100==0)&((controls["groups"]//100)%5!=0))
    lengths=np.load(folder/"lengths.npy",mmap_mode="r")
    order=fit[np.argsort(lengths[fit],kind="stable")]
    # 同时覆盖上限长度和短输入；较大拼批带来额外padding，也必须保持有效token输出。
    rows=np.unique(np.concatenate([order[-32:],order[np.linspace(0,len(order)-33,32,dtype=int)]]))
    if len(rows)!=64:raise ValueError("批量核验必须覆盖64个不同内部拟合输入。")
    path=FOLDER/"inner_lr3e-05_residual/epoch_2.pt"
    model=load_top(path).cuda().eval()
    # 两种形状均先预热；微型计时只用于本机调度，不当作完整新数据推理速度。
    for batch in [16,64]:extract(model,folder,rows=rows,batch_size=batch)
    values={};times={}
    torch.cuda.reset_peak_memory_stats()
    for batch in [16,64]:
        torch.cuda.synchronize();began=time.perf_counter()
        values[batch]=extract(model,folder,rows=rows,batch_size=batch)
        torch.cuda.synchronize();times[str(batch)]=time.perf_counter()-began
    logit_error=float(np.max(np.abs(values[16][0]-values[64][0])))
    cls_relative=float(np.linalg.norm(values[16][1]-values[64][1])/np.linalg.norm(values[16][1]))
    accepted=bool(logit_error<5e-5 and cls_relative<2e-5)
    # 数值用混合长度覆盖padding；性能另用同长度桶，接近正式按长度排序的大样本预测。
    # 混合64行全部补到128会额外浪费计算，不能据它推断完整排序数据的吞吐量。
    buckets=[order[lengths[order]==length][:64] for length in [8,16,32,64,128]]
    bucket_rows=np.concatenate([part for part in buckets if len(part)==64])
    if len(bucket_rows)<128:raise ValueError("同长度性能对照输入不足。")
    for batch in [16,64]:extract(model,folder,rows=bucket_rows,batch_size=batch)
    bucket_times={}
    for batch in [16,64]:
        torch.cuda.synchronize();began=time.perf_counter()
        for repeat in range(2):extract(model,folder,rows=bucket_rows,batch_size=batch)
        torch.cuda.synchronize();bucket_times[str(batch)]=(time.perf_counter()-began)/2
    selected_batch=64 if accepted and bucket_times["64"]<bucket_times["16"] else 16
    record={"accepted":accepted,"gpu_prediction_batch":selected_batch,"tested_gpu_prediction_batch":64,
        "larger_batch_enabled":selected_batch==64,"max_length":128,"rows":len(rows),
        "training_labels_read":False,"confirmation_labels_read":False,"training_micro_batch_unchanged":16,
        "checkpoint_sha256":sha256(path),"logit_max_abs_error":logit_error,"cls_relative_error":cls_relative,
        "warm_cached_microbenchmark_seconds":times,"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
        "length_bucket_rows":len(bucket_rows),"length_bucket_microbenchmark_seconds":bucket_times,
        "cpu_deployment_batch_unchanged":16,"long_context_prediction_batch_unchanged":16}
    write_json(OUT/"gpu_prediction_batch_audit.json",record);print(record,flush=True)
    if not accepted:raise RuntimeError("较大前向批量的数值核验未通过，继续使用16。")


if __name__=="__main__":main()
