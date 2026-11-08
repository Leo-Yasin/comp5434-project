"""从零实现类别条件的混合高斯-类别分布模型。

每个天气类别可包含多个潜在模式，不把普通天气、极端天气、宽分布样本
强行压成单峰。七个数值字段使用对角高斯，三个类别字段使用平滑类别分布。
EM、初始化、责任度、方差下界和概率计算全部由本项目实现。
"""
import numpy as np
from .data import NUMERIC,CATEGORICAL


def logsumexp(values,axis=1):
    maximum=values.max(axis=axis,keepdims=True)
    return np.squeeze(maximum+np.log(np.exp(values-maximum).sum(axis=axis,keepdims=True)),axis=axis)


class ConditionalMixture:
    def __init__(self,components=6,iterations=70,variance_floor=0.002,seed=5434):
        self.components,self.iterations=components,iterations
        self.variance_floor,self.seed=variance_floor,seed

    def transform(self,frame):
        numeric=(frame[NUMERIC].to_numpy(dtype=float)-self.location)/self.scale
        categorical=[]
        for col in CATEGORICAL:
            mapping={v:i for i,v in enumerate(self.categories[col])}
            categorical.append([mapping.get(str(v),len(mapping)) for v in frame[col]])
        return numeric,np.array(categorical,dtype=int).T

    @staticmethod
    def component_logprob(x,cat,state):
        mean,var=state["means"],state["variances"]
        distance=((x[:,None,:]-mean[None,:,:])**2/var[None,:,:]).sum(axis=2)
        result=-0.5*(distance+np.log(2*np.pi*var).sum(axis=1)[None,:])
        result+=np.log(state["weights"])[None,:]
        for j,density in enumerate(state["categorical"]):
            result+=np.log(density[:,cat[:,j]].T)
        return result

    def fit(self,frame,labels):
        raw=frame[NUMERIC].to_numpy(dtype=float)
        self.location,self.scale=raw.mean(0),np.maximum(raw.std(0),1e-6)
        self.categories={col:sorted(frame[col].astype(str).unique()) for col in CATEGORICAL}
        x,cat=self.transform(frame);labels=np.asarray(labels)
        self.prior=np.bincount(labels,minlength=5)[1:]/len(labels)
        self.states=[];self.history=[]
        rng=np.random.default_rng(self.seed)
        for label in range(1,5):
            xx,cc=x[labels==label],cat[labels==label]
            # k-means++ 只用于数值初始化；之后由 EM 联合学习数值与类别密度。
            centers=[xx[rng.integers(len(xx))]]
            for _ in range(1,self.components):
                d=np.min(np.sum((xx[:,None,:]-np.array(centers)[None,:,:])**2,axis=2),axis=1)
                chosen=rng.choice(len(xx),p=d/d.sum()) if d.sum()>1e-20 else rng.integers(len(xx))
                centers.append(xx[chosen])
            centers=np.array(centers)
            for _ in range(10):
                cluster=np.argmin(((xx[:,None,:]-centers[None,:,:])**2).sum(2),axis=1)
                for k in range(self.components):
                    if (cluster==k).any():centers[k]=xx[cluster==k].mean(0)
            responsibility=np.eye(self.components)[cluster]+0.01
            responsibility/=responsibility.sum(1,keepdims=True)
            previous=-np.inf
            for iteration in range(self.iterations):
                mass=responsibility.sum(0)+1e-10
                means=responsibility.T@xx/mass[:,None]
                second=responsibility.T@(xx**2)/mass[:,None]
                variance=np.maximum(second-means**2,self.variance_floor)
                densities=[]
                for j,col in enumerate(CATEGORICAL):
                    counts=np.column_stack([responsibility[cc[:,j]==v].sum(0)+0.25
                                            for v in range(len(self.categories[col])+1)])
                    densities.append(counts/counts.sum(1,keepdims=True))
                state={"means":means,"variances":variance,"weights":mass/mass.sum(),"categorical":densities}
                logits=self.component_logprob(xx,cc,state)
                normalizer=logsumexp(logits)
                responsibility=np.exp(logits-normalizer[:,None])
                objective=float(normalizer.mean())
                if iteration>10 and abs(objective-previous)<1e-5:break
                previous=objective
            self.states.append(state)
            self.history.append({"class":label,"iterations":iteration+1,"average_log_likelihood":objective})
        return self

    def predict_proba(self,frame):
        x,cat=self.transform(frame)
        logits=np.column_stack([logsumexp(self.component_logprob(x,cat,s)) for s in self.states])
        logits+=np.log(self.prior)
        return np.exp(logits-logsumexp(logits)[:,None])

    def to_dict(self):
        return {"location":self.location.tolist(),"scale":self.scale.tolist(),"categories":self.categories,
                "prior":self.prior.tolist(),"states":[{key:([v.tolist() for v in value] if key=="categorical" else value.tolist())
                                                         for key,value in state.items()} for state in self.states]}

    @classmethod
    def from_dict(cls,state):
        model=cls();model.location=np.array(state["location"]);model.scale=np.array(state["scale"])
        model.categories=state["categories"];model.prior=np.array(state["prior"])
        model.states=[{k:([np.array(v) for v in val] if k=="categorical" else np.array(val))
                       for k,val in item.items()} for item in state["states"]]
        return model
