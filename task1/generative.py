"""带共享背景成分的自主直方图生成式分类器。

每类主体密度为各特征直方图密度乘积；另用全体训练边际分布构成共享背景。
EM 只在带标签训练折中估计主体责任度，缓解少量宽分布样本污染条件密度。
该结构是需验证的统计假设，不将任意异常值直接删掉。
"""
import numpy as np


class BackgroundHistogram:
    def __init__(self,bins=32,alpha=0.5,iterations=12,background=True):
        self.bins,self.alpha,self.iterations,self.background=bins,alpha,iterations,background

    def transform(self,x):
        return np.column_stack([np.searchsorted(c,x[:,j],side="left") for j,c in enumerate(self.cuts)]).astype(int)

    def fit(self,x,labels):
        self.cuts=[]
        for column in x.T:
            u=np.unique(column)
            self.cuts.append((u[:-1]+u[1:])/2 if len(u)<=self.bins else
                             np.unique(np.quantile(column,np.linspace(0,1,self.bins+1)[1:-1])))
        z=self.transform(x)
        y=np.asarray(labels)-1
        self.prior=np.bincount(y,minlength=4)/len(y)
        self.background_density=[]
        for j,c in enumerate(self.cuts):
            counts=np.bincount(z[:,j],minlength=len(c)+1)+self.alpha
            self.background_density.append(counts/counts.sum())
        noise_log=sum(np.log(self.background_density[j][z[:,j]]) for j in range(z.shape[1]))
        responsibility=np.ones(len(y))
        self.rho=np.full(4,0.9)
        for iteration in range(self.iterations if self.background else 1):
            self.density=[]
            for j,c in enumerate(self.cuts):
                counts=np.array([np.bincount(z[y==k,j],weights=responsibility[y==k],minlength=len(c)+1)
                                 + self.alpha for k in range(4)])
                self.density.append(counts/counts.sum(axis=1,keepdims=True))
            signal_log=sum(np.log(self.density[j][y,z[:,j]]) for j in range(z.shape[1]))
            if self.background:
                ratio=np.clip(noise_log+np.log(1-self.rho[y])-signal_log-np.log(self.rho[y]),-50,50)
                responsibility=1/(1+np.exp(ratio))
                self.rho=np.array([np.clip(responsibility[y==k].mean(),0.6,0.99) for k in range(4)])
        return self

    def predict_proba(self,x):
        z=self.transform(x)
        score=np.tile(np.log(self.prior),(len(x),1))
        signal=sum(np.log(self.density[j][:,z[:,j]].T) for j in range(z.shape[1]))
        if self.background:
            noise=sum(np.log(self.background_density[j][z[:,j]]) for j in range(z.shape[1]))
            score += np.logaddexp(signal+np.log(self.rho),noise[:,None]+np.log(1-self.rho))
        else:
            score+=signal
        values=np.exp(score-score.max(axis=1,keepdims=True))
        return values/values.sum(axis=1,keepdims=True)

    def to_dict(self):
        return {"cuts":[x.tolist() for x in self.cuts],"density":[x.tolist() for x in self.density],
                "background_density":[x.tolist() for x in self.background_density],
                "rho":self.rho.tolist(),"prior":self.prior.tolist(),"background":self.background}

    @classmethod
    def from_dict(cls,state):
        model=cls(background=state["background"])
        for key in ["cuts","density","background_density"]:
            setattr(model,key,[np.asarray(v) for v in state[key]])
        model.rho,model.prior=np.array(state["rho"]),np.array(state["prior"])
        return model
