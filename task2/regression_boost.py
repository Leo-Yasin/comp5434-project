"""自主平方损失直方图提升树：为评分组合学习非线性残差。

梯度为 target-prediction，Hessian 为 1。每节点累计计数与梯度，按
G_L²/(n_L+lambda)+G_R²/(n_R+lambda)-G²/(n+lambda) 搜索二阶增益。
仅使用 NumPy，不调用第三方回归器。"""
import numpy as np


class ResidualHistogramBoost:
    def __init__(self,rounds=300,depth=2,learning_rate=.04,min_leaf=500,l2=500.,bins=64,seed=5434):
        self.rounds,self.depth,self.rate=rounds,depth,learning_rate
        self.min_leaf,self.l2,self.bins,self.seed=min_leaf,l2,bins,seed

    def encode(self,x):
        return np.column_stack([np.searchsorted(c,x[:,j],side="left") for j,c in enumerate(self.cuts)]).astype(np.uint8)

    def fit(self,x,target,initial):
        self.cuts=[]
        for column in x.T:
            unique=np.unique(column)
            cut=(unique[:-1]+unique[1:])/2 if len(unique)<=self.bins else np.unique(np.quantile(column,np.linspace(0,1,self.bins+1)[1:-1]))
            self.cuts.append(cut)
        encoded=self.encode(x);prediction=np.asarray(initial,dtype=float).copy()
        self.trees=[];self.history=[];rng=np.random.default_rng(self.seed)
        for round_ in range(self.rounds):
            gradient=target-prediction
            rows=np.sort(rng.choice(len(x),int(len(x)*.85),replace=False))
            tree=self.build(encoded,gradient,rows);self.trees.append(tree)
            prediction+=self.rate*self.tree_predict(tree,encoded)
            if (round_+1)%25==0:self.history.append({"round":round_+1,"pseudo_target_mse":float(np.mean((target-prediction)**2))})
        return self

    def build(self,x,gradient,rows):
        nodes=[]
        def visit(indices,depth):
            n=len(indices);g=gradient[indices].sum();index=len(nodes)
            nodes.append({"feature":-1,"value":float(np.clip(g/(n+self.l2),-2,2))})
            if depth>=self.depth or n<2*self.min_leaf:return index
            best_gain,feature,cut=1e-9,-1,-1
            for j,cuts in enumerate(self.cuts):
                if not len(cuts):continue
                code=x[indices,j]
                count=np.bincount(code,minlength=len(cuts)+1).cumsum()[:-1]
                total=np.bincount(code,weights=gradient[indices],minlength=len(cuts)+1).cumsum()[:-1]
                legal=(count>=self.min_leaf)&(n-count>=self.min_leaf)
                gain=total**2/(count+self.l2)+(g-total)**2/(n-count+self.l2)-g*g/(n+self.l2)
                gain[~legal]=-np.inf;at=int(np.argmax(gain))
                if gain[at]>best_gain:best_gain,feature,cut=float(gain[at]),j,at
            if feature>=0:
                left=x[indices,feature]<=cut
                li=visit(indices[left],depth+1);ri=visit(indices[~left],depth+1)
                nodes[index].update(feature=feature,threshold=cut,left=li,right=ri)
            return index
        visit(rows,0);return nodes

    @staticmethod
    def tree_predict(tree,x):
        out=np.empty(len(x))
        def visit(index,rows):
            if not len(rows):return
            node=tree[index]
            if node["feature"]<0:out[rows]=node["value"]
            else:
                mask=x[rows,node["feature"]]<=node["threshold"]
                visit(node["left"],rows[mask]);visit(node["right"],rows[~mask])
        visit(0,np.arange(len(x)));return out

    def predict(self,x,initial,rounds=None):
        encoded=self.encode(x);out=np.asarray(initial,dtype=float).copy()
        for tree in self.trees[:rounds]:out+=self.rate*self.tree_predict(tree,encoded)
        return out

    def to_dict(self):return {"cuts":[c.tolist() for c in self.cuts],"rate":self.rate,"trees":self.trees}

    @classmethod
    def from_dict(cls,state):
        model=cls();model.cuts=[np.array(c) for c in state["cuts"]];model.rate=state["rate"];model.trees=state["trees"]
        return model
