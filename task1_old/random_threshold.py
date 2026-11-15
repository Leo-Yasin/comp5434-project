"""自主随机阈值森林：不做 Bootstrap，每节点随机候选阈值后按 Gini 选择。

与旧森林逐个排序搜索全部阈值不同，随机阈值提供另一种偏差/方差取舍。
树节点采用现有 JSON 结构，便于使用独立加载器复现，不引入现成分类器。
"""
import numpy as np


class RandomThresholdForest:
    def __init__(self,trees=128,max_features=10,min_leaf=3,depth=25,seed=5434):
        self.n_trees,self.max_features,self.min_leaf,self.depth=trees,max_features,min_leaf,depth
        self.seed=seed

    def fit(self,x,labels):
        y=np.asarray(labels)-1;self.trees=[];rng=np.random.default_rng(self.seed)
        for _ in range(self.n_trees):
            nodes=[]
            def build(rows,depth):
                counts=np.bincount(y[rows],minlength=4);p=counts/len(rows)
                index=len(nodes)
                nodes.append({"feature":-1,"threshold":0.,"left":-1,"right":-1,"probability":p.tolist()})
                if depth>=self.depth or len(rows)<2*self.min_leaf or counts.max()==len(rows):return index
                values=x[rows];low,high=values.min(0),values.max(0)
                available=np.flatnonzero(high>low)
                if not len(available):return index
                features=np.sort(rng.choice(available,min(self.max_features,len(available)),replace=False))
                thresholds=rng.uniform(low[features],high[features])
                mask=values[:,features]<=thresholds[None,:]
                n_left=mask.sum(0);n_right=len(rows)-n_left
                valid=(n_left>=self.min_leaf)&(n_right>=self.min_leaf)
                if not valid.any():return index
                left=np.array([mask[y[rows]==k].sum(0) for k in range(4)],dtype=float).T
                right=counts-left
                score=np.full(len(features),np.inf)
                score[valid]=1-((left[valid]**2).sum(1)/n_left[valid]+(right[valid]**2).sum(1)/n_right[valid])/len(rows)
                candidate=int(np.argmin(score))
                if score[candidate]>=1-np.sum(p*p)-1e-12:return index
                li=build(rows[mask[:,candidate]],depth+1);ri=build(rows[~mask[:,candidate]],depth+1)
                nodes[index].update(feature=int(features[candidate]),threshold=float(thresholds[candidate]),left=li,right=ri)
                return index
            build(np.arange(len(y)),0)
            self.trees.append(nodes)
        return self

    def predict_proba(self,x):
        from .models import model_from_dict
        return model_from_dict(self.to_dict()).predict_proba(x)

    def to_dict(self):
        return {"type":"forest","trees":[{"type":"cart","nodes":nodes,"importance_raw":[]}
                                             for nodes in self.trees]}
