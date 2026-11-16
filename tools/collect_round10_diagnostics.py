"""不训练、不重新预测，只收集Round10复现诊断信息；不读取凭据或环境变量。

在朋友电脑上指定本次实际使用的三个Python、模型、测试文件、工作目录及预测CSV。
报告包含运行库版本、源码/模型SHA、阶段元数据和少量数值探针，不包含评论正文。
"""
import argparse,csv,hashlib,json,math,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def child(python,code,*args):
    # 每个角色用实际指定的解释器查询，避免把当前Python的依赖误当成其他环境的依赖。
    result=subprocess.run([str(python),"-X","utf8","-c",code,*map(str,args)],capture_output=True,text=True,encoding="utf-8")
    if result.returncode:return {"error":result.stderr[-3000:],"returncode":result.returncode}
    return json.loads(result.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["model","test","work-dir","prediction","semantic-python","english-python","output"]:parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--numeric-python",type=Path,default=Path(sys.executable))
    parser.add_argument("--reference",type=Path,help="已测评的Round10参考CSV；不包含测试真实标签。")
    args=parser.parse_args()
    packages=["numpy","pandas","scipy","scikit-learn","onnxruntime","onnxruntime-directml","openvino","torch","transformers","tokenizers","sentencepiece"]
    env_code="""import sys,platform,json,importlib.metadata as m
p=json.loads(sys.argv[1]);versions={}
for name in p:
 try:versions[name]=m.version(name)
 except m.PackageNotFoundError:versions[name]=None
r={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'packages':versions}
r['all_distributions']=sorted([{'name':d.metadata['Name'],'version':d.version} for d in m.distributions()],key=lambda d:str(d['name']))
try:
 import onnxruntime as ort
 r['available_providers_not_actual_selection']=ort.get_available_providers()
 r['onnxruntime_version']=ort.__version__
 r['onnxruntime_build_info']=ort.get_build_info()
except ImportError:pass
print(json.dumps(r))
"""
    environments={role:child(python,env_code,json.dumps(packages)) for role,python in [("numeric",args.numeric_python),("semantic",args.semantic_python),("english",args.english_python)]}
    report={"schema":1,"environments":environments,"test_sha256":sha(args.test),"source_sha256_lf":{},"stages":{},"model_integrity":{}}
    # Windows检出的CRLF与macOS的LF应视为相同源码；其余内容变化必须被记录。
    for path in sorted((ROOT/"task2").glob("*.py")):
        report["source_sha256_lf"][path.name]=hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    manifest=json.loads((args.model/"file_integrity.json").read_text(encoding="utf-8"));bad=[]
    for name,value in manifest.items():
        path=args.model/name
        if args.model.resolve() not in path.resolve().parents:raise ValueError("模型路径越界。")
        if not path.exists() or path.stat().st_size!=value["bytes"] or sha(path)!=value["sha256"]:bad.append(name)
    report["model_integrity"]={"manifest_sha256":sha(args.model/"file_integrity.json"),"checked":len(manifest),"mismatches":bad,
        "metadata":json.loads((args.model/"model.json").read_text(encoding="utf-8"))}
    # 选择前16个ID及相对参考CSV误差最大的16个ID，不把真实评分当作评估答案。
    reference=args.reference or ROOT/"submissions/20261114_task2_round10/task2_round10.csv"
    if not reference.exists() and args.reference is None:reference=ROOT/"submissions/latest/task2.csv"
    if args.reference is not None and not reference.exists():raise FileNotFoundError(reference)
    probes=[]
    if args.prediction.exists():
        with args.prediction.open(newline="",encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
        ratings=[float(row["rating"]) for row in rows]
        report["prediction_summary"]={"rows":len(rows),"unique_ids":len({row["id"] for row in rows}),
            "all_finite":all(math.isfinite(v) for v in ratings),"all_in_range":all(1 <= v <= 5 for v in ratings),
            "minimum":min(ratings) if ratings else None,"maximum":max(ratings) if ratings else None}
        probes=[int(row["id"]) for row in rows[:16]]
        if reference.exists():
            with reference.open(newline="",encoding="utf-8-sig") as stream:reference_rows=list(csv.DictReader(stream))
            old={row["id"]:float(row["rating"]) for row in reference_rows}
            if any(not math.isfinite(float(row["rating"])) for row in rows+reference_rows):raise ValueError("CSV包含非有限评分。")
            differences=[(abs(float(row["rating"])-old[row["id"]]),int(row["id"])) for row in rows if row["id"] in old]
            if differences:
                probes+= [id_ for _,id_ in sorted(differences,reverse=True)[:16]]
                report["prediction_comparison"]={"rows":len(rows),"matched_ids":len(differences),"mean_absolute_difference":sum(d for d,_ in differences)/len(differences),"maximum_absolute_difference":max(d for d,_ in differences),"prediction_sha256":sha(args.prediction),"reference_sha256":sha(reference)}
                report["prediction_comparison"].update({"reference_rows":len(reference_rows),"unique_prediction_ids":len({row['id'] for row in rows}),"unique_reference_ids":len(old),"id_order_equal":[r['id'] for r in rows]==[r['id'] for r in reference_rows],"root_mean_square_difference":math.sqrt(sum(d*d for d,_ in differences)/len(differences))})
    # 没有预测CSV时只采集已经完成的阶段；不读取正在写入的mmap数组。
    probes=list(dict.fromkeys(probes or list(range(16))));report["probe_ids"]=probes
    array_code="""import sys,json,numpy as np
from pathlib import Path
p=Path(sys.argv[1]);wanted=json.loads(sys.argv[2]);a=np.load(p);ids=a['id'];lookup={int(v):i for i,v in enumerate(ids)};pairs=[(v,lookup[v]) for v in wanted if v in lookup];r={'rows':len(ids),'probe_ids':[v for v,_ in pairs]}
for key in ['logits','prediction']:
 if key in a:r[key]=np.asarray(a[key])[[i for _,i in pairs]].tolist()
e=p.with_suffix('.embedding.npy')
if e.exists():r['cls']=np.load(e,mmap_mode='r')[[i for _,i in pairs]].tolist()
print(json.dumps(r))
"""
    for name in ["embedding","english","adapted","title"]:
        stage={}
        for suffix in [".state.json",".timing.json"]:
            path=args.work_dir/(name+suffix)
            if path.exists():stage[suffix]=json.loads(path.read_text(encoding="utf-8"))
        path=args.work_dir/(name+".npz")
        if path.exists():
            # 小探针用于量化差异，完整文件SHA用于判断相应数组是否被改动；
            # NPZ容器SHA不同本身不能代替对数组数值的比较。
            stage["artifact_sha256"]={path.name:sha(path)}
            embedding=path.with_suffix(".embedding.npy")
            if embedding.exists():stage["artifact_sha256"][embedding.name]=sha(embedding)
            stage["values"]=child(args.numeric_python,array_code,path,json.dumps(probes))
        report["stages"][name]=stage
    for name in ["tokens.json","state.json","timing.json"]:
        path=args.work_dir/"prefix"/name
        if path.exists():report.setdefault("prefix",{})[name]=json.loads(path.read_text(encoding="utf-8"))
    parent=args.prediction.with_suffix(".parent.npz")
    if parent.exists():report["parent_values"]=child(args.numeric_python,array_code,parent,json.dumps(probes))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("诊断报告完成",args.output,"模型SHA异常",len(bad),"探针ID",len(probes),flush=True)


if __name__=="__main__":main()
