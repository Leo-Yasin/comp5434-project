"""Dask 并行的一次 Huber-IRLS 加权词回归。

参考模型固定，a_i=min(1, delta/abs(y_i-reference_i))。然后自主用 PCG 求解
sum a_i*(X_i w+mean-y_i)^2 + w.T R w，只有原训练行进入权重/正规方程。
这是一轮固定重加权，不声称已迭代到 Huber 全局最优，也不直接将输出认作干净评分。
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from distributed import Client,LocalCluster
from .audit import ROOT,write_json
from .features import load_partition,TITLE_DIM,BODY_DIM
from .train import reduce_vectors,evaluate,hardware_info


def huber_weights(y,prediction,fit,delta):
    """验证行严格为0；不从验证评分计算训练权重。"""
    weight=np.zeros(len(y),dtype=float)
    residual=np.abs(y[fit]-prediction[fit])
    weight[fit]=np.minimum(1.,delta/np.maximum(residual,1e-12))
    return weight


def data_and_weights(prefix,mode,reference,mean,delta):
    x,y,fit,valid,ids=load_partition(prefix,mode,"hybrid")
    prediction=mean+x@reference
    return x,y,huber_weights(y,prediction,fit,delta),prediction,fit


def weighted_statistics(prefix,mode,reference,mean,delta):
    x,y,a,prediction,fit=data_and_weights(prefix,mode,reference,mean,delta)
    difference=np.zeros(len(y));difference[fit]=y[fit]-prediction[fit]
    gradient=np.asarray(x.T@(a*difference)).ravel()
    diagonal=np.asarray(x.multiply(x).T@a).ravel()
    return gradient,diagonal,float(a.sum()),int(fit.sum()),os.getpid()


def weighted_matvec(prefix,mode,direction,reference,mean,delta):
    x,y,a,prediction,fit=data_and_weights(prefix,mode,reference,mean,delta)
    return np.asarray(x.T@(a*(x@direction))).ravel()


def fit_weighted(client,parts,mode,reference,mean,delta,rounds,output,ridge=20.,intercept_index=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);began=time.perf_counter()
    ref=client.scatter([reference],broadcast=True,hash=False)[0]
    futures=[client.submit(weighted_statistics,p["prefix"],mode,ref,mean,delta,pure=False) for p in parts]
    gradient=diagonal=None;effective=0.;fit_rows=0;worker_pids=set()
    for future in futures:
        g,d,mass,n,pid=future.result()
        if gradient is None:gradient,diagonal=g.copy(),d.copy()
        else:gradient+=g;diagonal+=d
        effective+=mass;fit_rows+=n;worker_pids.add(pid);future.release()
    intercept_index=TITLE_DIM+BODY_DIM if intercept_index is None else intercept_index
    penalty=np.full(len(reference),ridge);penalty[intercept_index]=1e-6
    # 在参考参数处计算残差，不额外执行一遍 A@reference。
    residual=gradient-penalty*reference;diagonal+=penalty
    weights=reference.copy();z=residual/diagonal;direction=z.copy();rz=float(residual@z)
    norm=max(np.linalg.norm(residual),1e-30);history=[];setup_seconds=time.perf_counter()-began
    for iteration in range(1,rounds+1):
        started=time.perf_counter();pf=client.scatter([direction],broadcast=True,hash=False)[0]
        ap=reduce_vectors(client,parts,weighted_matvec,mode,pf,ref,mean,delta);pf.release()
        ap+=penalty*direction;denom=float(direction@ap)
        if denom<=0:raise RuntimeError("加权正规方程未保持正定。")
        alpha=rz/denom;weights+=alpha*direction;residual-=alpha*ap
        new_z=residual/diagonal;new_rz=float(residual@new_z)
        direction=new_z+(new_rz/max(rz,1e-300))*direction;rz=new_rz
        row={"iteration":iteration,"seconds":time.perf_counter()-started,
             "relative_residual_from_initial_weighted_gradient":float(np.linalg.norm(residual)/norm)}
        history.append(row);write_json(output/"history.json",history)
        print(f"robust/{mode} {iteration}/{rounds}: {row['seconds']:.2f}s",flush=True)
        if row["relative_residual_from_initial_weighted_gradient"]<1e-8:break
    ref.release();np.savez_compressed(output/"weights.npz",weights=weights,mean=mean)
    result={"mode":mode,"fixed_huber_delta":delta,"fixed_irls_updates":1,"iterations":len(history),
            "training_rows":fit_rows,"effective_weight_sum":effective,"actual_worker_pids":sorted(worker_pids),
            "setup_seconds":setup_seconds,"total_fit_seconds":time.perf_counter()-began,
            "mean_iteration_seconds":float(np.mean([r["seconds"] for r in history])),"source_model_is_reference":True}
    if mode=="split":
        metrics,records=evaluate(client,parts,"hybrid",weights,mean)
        ids,y,p,warm=records
        pd.DataFrame({"id":ids,"rating":y,"prediction":p}).to_csv(output/"validation_predictions.csv",index=False)
        result["raw_auxiliary_validation_metrics"]=metrics
    write_json(output/"summary.json",result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["split","full"],default="split")
    parser.add_argument("--rounds",type=int,default=12);parser.add_argument("--delta",type=float,default=1.)
    args=parser.parse_args()
    if args.delta<=0 or args.rounds<1:parser.error("阈值和迭代次数须为正。")
    out=ROOT/"task2/outputs/round4_20261105";out.mkdir(exist_ok=True)
    cache=json.loads((ROOT/"task2/cache/manifest.json").read_text(encoding="utf-8"))
    location=(ROOT/"task2/outputs/optimization_20261105/converged/hybrid/best_weights.npz" if args.mode=="split"
              else ROOT/"task2/outputs/optimization_20261105/full/hybrid/last_weights.npz")
    reference=np.load(location);weights=reference["weights"];mean=float(reference["mean"])
    with LocalCluster(n_workers=3,threads_per_worker=1,processes=True,dashboard_address=None,
                      memory_limit="1GB",silence_logs=40,local_directory=str(ROOT/"task2/tmp")) as cluster:
        with Client(cluster) as client:
            result=fit_weighted(client,cache["parts"],args.mode,weights,mean,args.delta,args.rounds,out/("robust_"+args.mode))
    write_json(out/("environment_"+args.mode+".json"),{"hardware":hardware_info(),"workers":3,
        "threads_per_worker":1,"framework":"Dask distributed","reference":str(location),"settings":vars(args)})


if __name__=="__main__":main()
