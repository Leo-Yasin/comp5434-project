"""第六轮自主预测头：分块统计的岭回归、噪声通道线性探针和MLP残差。

BERT只负责冻结表示。这里的归一化、目标函数、解析反向传播、dropout、Adam、
评分期望与融合均由项目代码实现，不调用第三方现成分类器。
"""
import time
import numpy as np
from scipy.optimize import minimize

from .noise_softmax import softmax, channel_objective


def normalization(features, fit):
    """只从训练行统计均值和方差；按批累计避免制造完整float64副本。"""
    rows=np.flatnonzero(fit);total=np.zeros(features.shape[1]);squares=total.copy()
    for begin in range(0,len(rows),4096):
        batch=np.asarray(features[rows[begin:begin+4096]],dtype=np.float64)
        total+=batch.sum(axis=0);squares+=(batch*batch).sum(axis=0)
    center=total/len(rows)
    scale=np.sqrt(np.maximum(squares/len(rows)-center*center,0))
    scale[scale<1e-6]=1.
    return center,scale


def fit_ridge(features, parent, rating, fit, epsilon, ridge):
    """解逆噪声评分残差的正规方程；不裁剪训练目标，只在最终评分处裁剪。"""
    began=time.perf_counter();center,scale=normalization(features,fit)
    rows=np.flatnonzero(fit);d=features.shape[1]+1
    gram=np.zeros((d,d));rhs=np.zeros(d)
    for begin in range(0,len(rows),4096):
        at=rows[begin:begin+4096]
        x=np.column_stack([np.ones(len(at)),(features[at]-center)/scale])
        target=(rating[at]-3*epsilon)/(1-epsilon)-parent[at]
        gram+=x.T@x;rhs+=x.T@target
    penalty=np.full(d,ridge);penalty[0]=1.
    coefficient=np.linalg.solve(gram+np.diag(penalty),rhs)
    return {"center":center,"scale":scale,"coefficient":coefficient},{
        "training_rows":len(rows),"seconds":time.perf_counter()-began,"ridge":ridge}


def ridge_predict(features,parent,state):
    out=np.asarray(parent,dtype=np.float64).copy()
    for at in range(0,len(features),4096):
        x=(features[at:at+4096]-state["center"])/state["scale"]
        out[at:at+len(x)]+=state["coefficient"][0]+x@state["coefficient"][1:]
    return out


def fit_linear(features,offset,rating,fit,epsilon,ridge,iterations=500):
    """低内存线性探针；通道损失及其解析梯度复用已经有限差分检验的实现。"""
    began=time.perf_counter();center,scale=normalization(features,fit)
    rows=np.flatnonzero(fit)
    x=np.column_stack([np.ones(len(rows)),(features[rows]-center)/scale])
    y=rating[rows].astype(int)-1
    result=minimize(channel_objective,np.zeros(x.shape[1]*5),
        args=(x,offset[rows],y,epsilon,ridge),method="L-BFGS-B",jac=True,
        options={"maxiter":iterations,"ftol":1e-9,"gtol":1e-5,"maxls":30})
    state={"center":center,"scale":scale,"weights":result.x.reshape(x.shape[1],5)}
    return state,{"training_rows":len(rows),"seconds":time.perf_counter()-began,
        "iterations":int(result.nit),"success":bool(result.success),"message":str(result.message),
        "ridge":ridge,"objective":float(result.fun)}


def linear_logits(features,offset,state):
    out=np.asarray(offset,dtype=np.float64).copy()
    for at in range(0,len(features),4096):
        x=(features[at:at+4096]-state["center"])/state["scale"]
        out[at:at+len(x)]+=state["weights"][0]+x@state["weights"][1:]
    return out


def noise_loss_gradient(logits,rating,epsilon):
    """平均负对数似然及对logits的梯度：责任权重w=(1-eps)*q_y/r_y。"""
    probability=softmax(logits);index=rating.astype(int)-1;row=np.arange(len(index))
    clean=probability[row,index];observed=(1-epsilon)*clean+epsilon/5
    gradient=probability.copy();gradient[row,index]-=1.
    gradient*=(((1-epsilon)*clean/np.maximum(observed,1e-15))/len(index))[:,None]
    return float(-np.log(np.maximum(observed,1e-15)).mean()),gradient


def mlp_loss_gradients(x,offset,rating,parameters,epsilon,l2,mask=None):
    """显式链式法则；mask已包含1/keep缩放，None用于无dropout的数学检查。"""
    h=np.tanh(x@parameters["w1"]+parameters["b1"])
    effective=h if mask is None else h*mask
    logits=offset+effective@parameters["w2"]+parameters["b2"]
    loss,g=noise_loss_gradient(logits,rating,epsilon)
    loss+=.5*l2*(np.sum(parameters["w1"]**2)+np.sum(parameters["w2"]**2))
    dw2=effective.T@g+l2*parameters["w2"];db2=g.sum(axis=0)
    dh=g@parameters["w2"].T
    if mask is not None:dh*=mask
    dh*=1-h*h
    return loss,{"w1":x.T@dh+l2*parameters["w1"],"b1":dh.sum(axis=0),"w2":dw2,"b2":db2}


def fit_mlp(features,offset,rating,fit,epsilon,anchor,epochs=12,hidden=64,
            dropout=.2,learning_rate=.001,l2=.001,seed=5434):
    """在线性探针上学习64维tanh残差；训练顺序和dropout均由固定随机种子生成。

    anchor固定不再优化，MLP只学习额外残差。只读取fit行的标签；不以内外验证标签更新梯度。
    保留3/6/12轮快照供既定内部比较，最终用被冻结轮数从头重训。
    """
    began=time.perf_counter();rng=np.random.default_rng(seed)
    rows=np.flatnonzero(fit)
    # float32控制内存；训练/独立推理使用同一标准化和锚点logits。
    x=((features[rows]-anchor["center"])/anchor["scale"]).astype(np.float32)
    anchored=linear_logits(features[rows],offset[rows],anchor).astype(np.float32)
    y=rating[rows]
    parameters={"w1":rng.normal(0,1/np.sqrt(x.shape[1]),size=(x.shape[1],hidden)).astype(np.float32),
        "b1":np.zeros(hidden,np.float32),"w2":np.zeros((hidden,5),np.float32),"b2":np.zeros(5,np.float32)}
    first={k:np.zeros_like(v) for k,v in parameters.items()}
    second={k:np.zeros_like(v) for k,v in parameters.items()}
    snapshots={};history=[];step=0
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(rows));losses=[];started=time.perf_counter()
        for at in range(0,len(rows),512):
            batch=order[at:at+512]
            mask=(rng.random((len(batch),hidden))>=dropout).astype(np.float32)/(1-dropout)
            loss,gradient=mlp_loss_gradients(x[batch],anchored[batch],y[batch],parameters,epsilon,l2,mask)
            # 全局梯度裁剪与标准Adam偏差修正，不能把原始梯度裁剪成类别硬标签。
            norm=np.sqrt(sum(float(np.sum(v*v)) for v in gradient.values()));factor=min(1.,5./max(norm,1e-12))
            step+=1
            for name in parameters:
                g=(gradient[name]*factor).astype(np.float32)
                first[name]=.9*first[name]+.1*g;second[name]=.999*second[name]+.001*g*g
                parameters[name]-=learning_rate*(first[name]/(1-.9**step))/(np.sqrt(second[name]/(1-.999**step))+1e-8)
            losses.append(loss)
        history.append({"epoch":epoch,"seconds":time.perf_counter()-started,
            "mean_minibatch_training_loss":float(np.mean(losses))})
        if epoch in [3,6,12] or epoch==epochs:
            snapshots[epoch]={**{k:v.copy() for k,v in parameters.items()},
                **{"anchor_"+k:v.copy() for k,v in anchor.items()}}
            print("MLP训练",epoch,history[-1],flush=True)
    return snapshots,{"seconds":time.perf_counter()-began,"training_rows":len(rows),
        "epochs":epochs,"hidden":hidden,"dropout":dropout,"learning_rate":learning_rate,
        "l2":l2,"seed":seed,"history":history}


def mlp_predict(features,offset,state):
    anchor={k:state["anchor_"+k] for k in ["center","scale","weights"]}
    out=np.empty(len(features))
    for at in range(0,len(features),2048):
        batch=features[at:at+2048]
        x=((batch-anchor["center"])/anchor["scale"]).astype(np.float32)
        logits=linear_logits(batch,offset[at:at+len(batch)],anchor).astype(np.float32)
        logits+=np.tanh(x@state["w1"]+state["b1"])@state["w2"]+state["b2"]
        out[at:at+len(batch)]=softmax(logits)@np.arange(1.,6.)
    return out


def predict_head(features,parent,offset,state,choice):
    if choice["family"]=="ridge":candidate=ridge_predict(features,parent,state)
    elif choice["family"]=="linear":candidate=softmax(linear_logits(features,offset,state))@np.arange(1.,6.)
    elif choice["family"]=="mlp":candidate=mlp_predict(features,offset,state)
    elif choice["family"]=="parent":return np.asarray(parent).copy()
    else:raise ValueError("未知第六轮预测头。")
    return np.clip(parent+choice["alpha"]*(candidate-parent),1,5)
