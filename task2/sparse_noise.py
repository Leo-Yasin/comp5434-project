"""全量稀疏五星噪声通道分类：Dask并行计算，自主损失与梯度。

原词模型只学习一个评分均值。本模型学习五个潜在类别，显式建模25%左右的
随机评分替换；用通用L-BFGS数值优化器优化本项目定义的目标，不调用现成分类器。
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from distributed import Client, LocalCluster
from .audit import ROOT, write_json
from .features import load_partition, DIM, TITLE_DIM, BODY_DIM
from .noise_softmax import softmax
from .embedding_heads import noise_loss_gradient

OUT=ROOT/"task2/outputs/round7_local_20261114"


def training_partition(prefix,mode,representation=None):
    """可选TF-IDF缓存；省略时逐字保持上一轮原表示和测试接口。"""
    if representation is None:return load_partition(prefix,mode,"hybrid")
    from .tfidf_sparse import partition
    return partition(prefix,mode,representation)


def batch_diagonal(prefixes,mode,representation=None):
    """无标签Jacobi尺度：稀疏列平方和用于改善不同词频带来的病态曲率。"""
    diagonal=np.zeros(DIM);rows=0
    for prefix in prefixes:
        x,y,fit,valid,ids=training_partition(prefix,mode,representation)
        diagonal+=np.asarray(x.multiply(x).T@fit.astype(float)).ravel();rows+=int(fit.sum())
    return diagonal,rows


def batch_objective(prefixes, mode, weights, bias, epsilon,representation=None):
    """一个真实Dask工作进程顺序遍历自己的分区，只归约训练行。

    先计算所有行的logits，再将验证行梯度置0；损失只对训练行求和。
    因此修改验证标签不会影响本函数结果。
    """
    gradient=np.zeros_like(weights);gb=np.zeros(5);loss=0.;rows=0
    for prefix in prefixes:
        x,y,fit,valid,ids=training_partition(prefix,mode,representation)
        logits=x@weights+bias;q=softmax(logits)
        row=np.arange(len(y));label=y.astype(int)-1
        py=q[row,label];observed=(1-epsilon)*py+epsilon/5
        loss-=np.log(np.maximum(observed[fit],1e-15)).sum()
        q[row,label]-=1
        q*=((1-epsilon)*py/np.maximum(observed,1e-15)*fit)[:,None]
        gradient+=x.T@q;gb+=q.sum(axis=0);rows+=int(fit.sum())
    return float(loss),gradient,gb,rows,os.getpid()


def initialization(mode,epsilon):
    """用已训练均值回归构造有序高斯先验，避免从随机类别开始全量训练。

    softmax(-.75*(star-mean(x))²) 中与类别无关的mean(x)²会抵消，
    因而可以精确写成稀疏线性logits，作为新的优化起点。
    """
    path=ROOT/"task2/outputs/optimization_20261105"/("converged/hybrid/best_weights.npz" if mode=="split" else "full/hybrid/last_weights.npz")
    state=np.load(path);stars=np.arange(-2.,3.)
    weights=1.5*state["weights"][:,None]*stars[None,:]/(1-epsilon)
    mean=(float(state["mean"])-3*epsilon)/(1-epsilon)-3
    bias=1.5*stars*mean-.75*stars**2
    return weights,bias


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["split","full"],required=True)
    parser.add_argument("--iterations",type=int,default=60);parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--scaled",action="store_true");parser.add_argument("--resume",type=str)
    parser.add_argument("--output",type=Path);parser.add_argument("--ridge",type=float,default=50.)
    parser.add_argument("--idf-cache",type=Path)
    args=parser.parse_args();folder=args.output or OUT/("sparse_pilot" if args.pilot else ("sparse_scaled_" if args.scaled else "sparse_")+args.mode)
    folder.mkdir(parents=True,exist_ok=False)
    parts=json.loads((ROOT/"task2/cache/manifest.json").read_text(encoding="utf-8"))["parts"]
    if args.pilot:parts=parts[:3]
    eps=float(np.load(ROOT/"task2/outputs/round3_20261105/development.npz")["epsilon"])
    weights,bias=initialization(args.mode,eps);start=np.concatenate([weights.ravel(),bias])
    if args.resume:
        source=np.load(args.resume);weights=source["weights"];bias=source["bias"]
    if args.ridge<=0:raise ValueError("L2须为正。")
    representation=json.loads((args.idf_cache/"manifest.json").read_text(encoding="utf-8")) if args.idf_cache else None
    penalty=np.full((DIM,5),args.ridge);penalty[TITLE_DIM+BODY_DIM]=0.
    history=[];iterations=[];began=time.perf_counter();worker_pids=set()
    with LocalCluster(n_workers=3,threads_per_worker=1,processes=True,dashboard_address=None,
        memory_limit="1500MB",silence_logs=40,local_directory=str(ROOT/"task2/tmp")) as cluster:
        with Client(cluster) as client:
            scale=np.ones((DIM,1));bias_scale=1.
            if args.scaled:
                futures=[client.submit(batch_diagonal,[p["prefix"] for p in parts[j::3]],args.mode,representation,pure=False) for j in range(3)]
                diagonal=np.zeros(DIM);rows=0
                for future in futures:
                    value,n=future.result();diagonal+=value;rows+=n;future.release()
                scale=np.sqrt(np.maximum(.2*diagonal+penalty[:,0],1.))[:,None]
                bias_scale=np.sqrt(.2*rows)
                np.savez_compressed(folder/"optimizer_scale.npz",scale=scale,bias_scale=bias_scale)
            start=np.concatenate([(weights*scale).ravel(),bias*bias_scale])
            def objective(flat):
                started=time.perf_counter();w=flat[:-5].reshape(DIM,5)/scale;b=flat[-5:]/bias_scale
                wf=client.scatter([w],broadcast=True,hash=False)[0]
                futures=[client.submit(batch_objective,[p["prefix"] for p in parts[j::3]],args.mode,wf,b,eps,representation,pure=False) for j in range(3)]
                loss=0.;g=np.zeros_like(w);gb=np.zeros(5);rows=0
                for future in futures:
                    value,dw,db,n,pid=future.result();loss+=value;g+=dw;gb+=db;rows+=n;worker_pids.add(pid);future.release()
                wf.release();loss+=.5*np.sum(penalty*w*w);g+=penalty*w
                record={"evaluation":len(history)+1,"loss":float(loss),"rows":rows,"seconds":time.perf_counter()-started}
                history.append(record)
                if len(history)%5==1:print("稀疏五星目标",args.mode,record,flush=True)
                return loss,np.concatenate([(g/scale).ravel(),gb/bias_scale])
            if args.pilot:
                objective(start)
                result={"pilot":True,"history":history,"worker_pids":sorted(worker_pids)}
            else:
                def callback(flat):
                    iterations.append(len(history))
                    if len(iterations)%10==0:
                        np.savez_compressed(folder/f"checkpoint_{len(iterations)}.npz",weights=flat[:-5].reshape(DIM,5)/scale,bias=flat[-5:]/bias_scale,epsilon=eps)
                        write_json(folder/"progress.json",{"iterations":len(iterations),"history":history})
                fit=minimize(objective,start,jac=True,method="L-BFGS-B",callback=callback,
                    options={"maxiter":args.iterations,"maxcor":7,"ftol":1e-8,"gtol":1e-4,"maxls":20})
                np.savez_compressed(folder/"weights.npz",weights=fit.x[:-5].reshape(DIM,5)/scale,bias=fit.x[-5:]/bias_scale,epsilon=eps)
                result={"success":bool(fit.success),"message":str(fit.message),"iterations":int(fit.nit),
                        "objective_evaluations":len(history),"history":history,"worker_pids":sorted(worker_pids)}
    result.update(seconds=time.perf_counter()-began,scaled=args.scaled,resume=args.resume,ridge=args.ridge,representation=representation);write_json(folder/"training.json",result)


if __name__=="__main__":main()
