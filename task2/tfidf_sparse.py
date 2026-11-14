"""第八轮分块TF-IDF表示：保留标题/正文分块L2归一化和原数值特征。

IDF仅由原外层训练行的文档频率估计，验证/测试文本不参与估计。
新浮点缓存放到用户缓存盘，原稀疏索引与原模型文件完全不改。
"""
from collections import OrderedDict
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from .audit import ROOT,write_json
from .features import TITLE_DIM,BODY_DIM,TEXT_DIM

CACHE=OrderedDict()
WORD_DIM=TITLE_DIM+BODY_DIM


def transform(text,idf):
    """先乘IDF，再分别归一化标题与正文；其余8个基本数值列逐值保留。

    哈希列可能有正负号，计算平方范数而不是把负号当成无效词。
    标题归一化后恢复1.5权重，正文恢复1.0权重。
    """
    x=text.tocsr(copy=True)
    if len(idf)!=WORD_DIM:raise ValueError("IDF维度不正确。")
    words=x.indices<WORD_DIM
    x.data[words]*=idf[x.indices[words]].astype(x.data.dtype)
    rows=np.repeat(np.arange(x.shape[0],dtype=np.int32),np.diff(x.indptr))
    for mask,weight in [(x.indices<TITLE_DIM,1.5),((x.indices>=TITLE_DIM)&words,1.)]:
        norm=np.sqrt(np.bincount(rows[mask],weights=np.asarray(x.data[mask],dtype=np.float64)**2,minlength=x.shape[0]))
        factors=np.divide(weight,norm,out=np.zeros_like(norm),where=norm>0)
        x.data[mask]*=factors[rows[mask]].astype(x.data.dtype)
    return x


def partition(prefix,mode,config):
    """读取新数据值和旧只读索引；元数据仍按原协议交叉拟合。"""
    key=(prefix,mode,config["folder"])
    if key in CACHE:return CACHE[key]
    meta=dict(np.load(prefix+".meta.npz"));g=meta["groups"];valid=g%np.uint64(100)<2
    values=Path(config["folder"])/(Path(prefix).name+".data.npy")
    text=sparse.csr_matrix((np.load(values,mmap_mode="r"),np.load(prefix+".indices.npy",mmap_mode="r"),
        np.load(prefix+".indptr.npy",mmap_mode="r")),shape=(len(g),TEXT_DIM),copy=False)
    numeric=np.load(prefix+f".{mode}.numeric.npy",mmap_mode="r")
    x=sparse.hstack([text,sparse.csr_matrix(numeric)],format="csr",dtype=np.float32)
    result=(x,meta["rating"],~valid if mode=="split" else np.ones(len(g),dtype=bool),valid,meta["id"])
    CACHE[key]=result
    while len(CACHE)>2:CACHE.popitem(last=False)
    return result


def prepare(folder):
    """两遍流式扫描：第一遍计算训练DF，第二遍保存新数值，不复制索引数组。"""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/"task2/cache/manifest.json").read_text(encoding="utf-8"))
    df=np.zeros(WORD_DIM,dtype=np.int64);n=0;started=time.perf_counter()
    for part in manifest["parts"]:
        prefix=part["prefix"];g=np.load(prefix+".meta.npz")["groups"];fit=g%np.uint64(100)>=2
        x=sparse.csr_matrix((np.load(prefix+".data.npy",mmap_mode="r"),np.load(prefix+".indices.npy",mmap_mode="r"),
            np.load(prefix+".indptr.npy",mmap_mode="r")),shape=(len(g),TEXT_DIM),copy=False)[fit,:WORD_DIM]
        # 原CSR每个哈希列在一行最多一个元素；消掉碰撞抵消为0的元素后计数。
        x.eliminate_zeros();df+=np.bincount(x.indices,minlength=WORD_DIM);n+=int(fit.sum())
    idf=(np.log((1+n)/(1+df))+1).astype(np.float32)
    np.save(folder/"idf.npy",idf)
    for part in manifest["parts"]:
        prefix=part["prefix"]
        x=sparse.csr_matrix((np.load(prefix+".data.npy",mmap_mode="r"),np.load(prefix+".indices.npy",mmap_mode="r"),
            np.load(prefix+".indptr.npy",mmap_mode="r")),shape=(part["rows"],TEXT_DIM),copy=False)
        changed=transform(x,idf)
        np.save(folder/(Path(prefix).name+".data.npy"),changed.data)
        if part["number"]%25==0:print("TF-IDF缓存",part["number"],"seconds",time.perf_counter()-started,flush=True)
    write_json(folder/"manifest.json",{"folder":str(folder.resolve()),"rows":manifest["rows"],"idf_fit_rows":n,
        "mode":"split_idf_used_for_validation_and_test","title_weight":1.5,"separate_l2":True,
        "seconds":time.perf_counter()-started})


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--folder",type=Path,required=True);args=p.parse_args();prepare(args.folder)
