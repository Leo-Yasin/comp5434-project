"""用 Dask 实际构建重复文本和用户-商品交互键，不改变旧稀疏特征缓存。

统一小写/空白的文本仍是输入特征，不能被视为真实标签。
键哈希不包含 rating；聚合时由后续脚本明确排除全部外层验证评分。
"""
from pathlib import Path
import argparse,json,os,time
import numpy as np
import pandas as pd
from distributed import Client,LocalCluster
from .audit import ROOT,CSV_OPTIONS,group_hash,normalize,write_json

KEYS=["title","comment","text_pair","user_product","user_parent"]


def make_keys(frame):
    """规范化只依赖该记录文本，不从测试集学习词表或缩放参数。"""
    frame=normalize(frame)
    title=frame.title.str.lower().str.replace(r"\s+"," ",regex=True).str.strip()
    comment=frame.comment.str.lower().str.replace(r"\s+"," ",regex=True).str.strip()
    pairs={"title":title,"comment":comment,
           "text_pair":pd.DataFrame({"title":title,"comment":comment}),
           "user_product":frame[["user_id","prod_id"]],
           "user_parent":frame[["user_id","parent_prod_id"]]}
    result={key:pd.util.hash_pandas_object(value,index=False).to_numpy(dtype=np.uint64)
            for key,value in pairs.items()}
    # 空标题/过短正文形成的公共键没有足够信息，单独保存有效性，推理时回退。
    result["valid_title"]=(title.str.len()>=3).to_numpy()
    result["valid_comment"]=(comment.str.len()>=12).to_numpy()
    result["valid_text_pair"]=(comment.str.len()>=12).to_numpy()
    return result


def partition(frame,number,out):
    keys=make_keys(frame)
    keys.update(groups=group_hash(normalize(frame)),rating=frame.rating.to_numpy(dtype=np.float32))
    target=Path(out)/f"part_{number:04d}.npz"
    np.savez_compressed(target,**keys)
    return {"number":number,"rows":len(frame),"path":str(target.resolve()),"pid":os.getpid()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    out=ROOT/"task2/outputs/round2_20261105";out.mkdir(exist_ok=False)
    shards=out/"keys";shards.mkdir();began=time.perf_counter()
    records=[];pending=[]
    with LocalCluster(n_workers=args.workers,threads_per_worker=1,processes=True,
                      dashboard_address=None,memory_limit="1GB",silence_logs=40,
                      local_directory=str(ROOT/"task2/tmp")) as cluster:
        with Client(cluster) as client:
            for number,frame in enumerate(pd.read_csv(ROOT/"comp-5434-20261-project-task-2/train.csv",
                                                      chunksize=100000,**CSV_OPTIONS)):
                future=client.scatter(frame,hash=False)
                pending.append(client.submit(partition,future,number,str(shards),pure=False))
                if len(pending)>=args.workers*2:
                    record=pending.pop(0).result();records.append(record)
                    if record["number"]%10==0:print(f"新增统计键完成 {record['number']+1} 块",flush=True)
            records.extend(f.result() for f in pending)
    records.sort(key=lambda r:r["number"])
    test=pd.read_csv(ROOT/"comp-5434-20261-project-task-2/test.csv",**CSV_OPTIONS)
    np.savez_compressed(out/"test_keys.npz",id=test.id.to_numpy(),**make_keys(test))
    write_json(out/"keys_manifest.json",{"parts":records,"workers":args.workers,
        "actual_worker_pids":sorted({r["pid"] for r in records}),"seconds":time.perf_counter()-began})


if __name__=="__main__":main()
