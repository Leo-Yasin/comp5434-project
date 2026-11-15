"""自主类别容量约束的最小代价分配。

在假设预测集类别比例接近训练集时，按训练先验分配整数配额，最大化
sum_i p(i, assigned_class)。不读取预测集标签，也不改变训练模型。
先修复超额类别，再消除负费用循环；无负循环是该运输问题的最优性条件。
"""
import heapq
import itertools
import numpy as np


def prior_quotas(n,prior):
    expected=n*np.asarray(prior)/np.sum(prior)
    quotas=np.floor(expected).astype(int)
    remainder=n-int(quotas.sum())
    order=np.argsort(-(expected-quotas),kind="stable")
    quotas[order[:remainder]]+=1
    return quotas


def constrained_assignment(scores,quotas):
    scores=np.asarray(scores,dtype=float);quotas=np.asarray(quotas,dtype=int)
    n,k=scores.shape
    if quotas.shape!=(k,) or quotas.sum()!=n or (quotas<0).any() or not np.isfinite(scores).all():
        raise ValueError("配额或概率矩阵不合法。")
    labels=scores.argmax(1);generation=np.zeros(n,dtype=int)
    heaps={(a,b):[] for a in range(k) for b in range(k) if a!=b}
    def insert(i):
        a=int(labels[i])
        for b in range(k):
            if a!=b:heapq.heappush(heaps[a,b],(float(scores[i,a]-scores[i,b]),i,int(generation[i])))
    for i in range(n):insert(i)
    def edges():
        result={}
        for (a,b),heap in heaps.items():
            while heap and (labels[heap[0][1]]!=a or generation[heap[0][1]]!=heap[0][2]):heapq.heappop(heap)
            if heap:result[a,b]=(heap[0][0],heap[0][1])
        return result
    def change(path,edge):
        # 路径中源类别互不相同，因此所选记录也互不相同，先收集后执行。
        moves=[(edge[a,b][1],b) for a,b in zip(path[:-1],path[1:])]
        for i,b in moves:labels[i]=b;generation[i]+=1;insert(i)
    counts=np.bincount(labels,minlength=k)
    while (counts!=quotas).any():
        edge=edges();best=None
        for a in np.flatnonzero(counts>quotas):
            for b in np.flatnonzero(counts<quotas):
                others=[c for c in range(k) if c not in [a,b]]
                for length in range(len(others)+1):
                    for middle in itertools.permutations(others,length):
                        path=(int(a),)+middle+(int(b),)
                        if all((u,v) in edge for u,v in zip(path[:-1],path[1:])):
                            cost=sum(edge[u,v][0] for u,v in zip(path[:-1],path[1:]))
                            if best is None or cost<best[0]-1e-12:best=(cost,path)
        if best is None:raise RuntimeError("容量修复没有可行路径。")
        change(best[1],edge);counts[best[1][0]]-=1;counts[best[1][-1]]+=1
    # 固定配额后改善剩余负循环，避免贪心修复只获得局部解。
    for iteration in range(100000):
        edge=edges();best=None
        for length in range(2,k+1):
            for cycle in itertools.permutations(range(k),length):
                if cycle[0]!=min(cycle):continue
                path=cycle+(cycle[0],)
                if all((a,b) in edge for a,b in zip(path[:-1],path[1:])):
                    cost=sum(edge[a,b][0] for a,b in zip(path[:-1],path[1:]))
                    if cost<-1e-12 and (best is None or cost<best[0]):best=(cost,path)
        if best is None:break
        change(best[1],edge)
    else:raise RuntimeError("循环改善未收敛，拒绝输出近似结果。")
    if not np.array_equal(np.bincount(labels,minlength=k),quotas):raise RuntimeError("最终配额不符。")
    return labels
