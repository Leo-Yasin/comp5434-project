"""用户确认不再训练后，清除可重建缓存/重复文件，保留第十轮完整推理包。

每个删除目标先验证绝对路径与明确允许范围，禁止触及最终模型或其父目录。
保留原始数据、所有提交CSV、源码、文档、CPU推理环境及模型校验清单。
"""
import hashlib,json,shutil,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path.home()/".cache"
KEEP=CACHE/"comp5434-title10/model"
LOG=ROOT/"docs/cleanup-20261115"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    LOG.mkdir(exist_ok=True)
    if (LOG/"result.json").exists():raise FileExistsError("本次清理已完成，不重复删除。")
    manifest=json.loads((KEEP/"file_integrity.json").read_text(encoding="utf-8"))
    def verify_model():
        for name,item in manifest.items():
            p=KEEP/name
            if p.stat().st_size!=item["bytes"] or sha(p)!=item["sha256"]:raise ValueError("最终模型不完整："+name)
    verify_model();print("最终模型校验通过",len(manifest)+1,"文件",flush=True)
    protected=[p for d in [ROOT/"submissions",ROOT/"comp-5434-20261-project-task-1",ROOT/"comp-5434-20261-project-task-2"] for p in d.rglob("*") if p.is_file()]
    snapshot={str(p):sha(p) for p in protected};snapshot[str(KEEP/"file_integrity.json")]=sha(KEEP/"file_integrity.json")
    (LOG/"protected_sha256.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding="utf-8")
    # 上传和本地验证的元数据留档；压缩包/分卷已由GitHub逐附件SHA验证并公开备份。
    receipts=LOG/"upload-receipts";receipts.mkdir(exist_ok=True)
    release=CACHE/"comp5434-title10/github-release"
    published=json.loads((release/"published_release.json").read_text(encoding="utf-8"))
    if not published["all_asset_digests_verified"]:raise ValueError("未确认模型上传完整，不删除本地上传包。")
    for p in release.iterdir():
        if p.suffix in [".json",".sha256"]:shutil.copy2(p,receipts/p.name)
    targets=[CACHE/"comp5434-local9",CACHE/"comp5434-english/model",ROOT/"task2/cache",ROOT/"task2/audit",ROOT/"task2/audit_compatible"]
    title=CACHE/"comp5434-title10"
    # 只保留当前model；上传副本、测试解压副本、训练向量和推理缓存均可重建。
    targets.extend(p for p in title.iterdir() if p.name!="model")
    extensions={".npy",".npz",".onnx",".bin",".pt",".safetensors",".pkl",".joblib",".zip",".7z"}
    targets.extend(p for p in (ROOT/"task2/outputs").rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    targets.extend(p for p in (ROOT/"task2").rglob("__pycache__") if p.is_dir())
    allowed=[CACHE/"comp5434-local9",CACHE/"comp5434-english/model",title,ROOT/"task2/cache",ROOT/"task2/audit",ROOT/"task2/audit_compatible",ROOT/"task2/outputs",ROOT/"task2"]
    unique=[]
    for p in sorted(set(targets),key=lambda v:len(v.parts)):
        absolute=p.absolute();resolved=p.resolve()
        if not p.exists():continue
        if resolved!=absolute:raise ValueError("拒绝重解析/符号链接目标："+str(p))
        if resolved==KEEP or resolved in KEEP.parents or KEEP in resolved.parents:raise ValueError("拒绝触及最终模型："+str(p))
        if not any(resolved==r.resolve() or r.resolve() in resolved.parents for r in allowed):raise ValueError("删除路径越界："+str(p))
        if any(parent in resolved.parents for parent in unique):continue
        unique.append(resolved)
    plan=[]
    for p in unique:
        size=sum(v.stat().st_size for v in p.rglob("*") if v.is_file()) if p.is_dir() else p.stat().st_size
        plan.append({"path":str(p),"directory":p.is_dir(),"logical_bytes":size})
    (LOG/"deletion_plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    before={drive:shutil.disk_usage(drive+":/").free for drive in ["C","F"]}
    for i,item in enumerate(plan,1):
        p=Path(item["path"])
        # 删除前再次确认已解析绝对路径仍为计划中的精确目标。
        if p.resolve()!=p or p==KEEP or p in KEEP.parents:raise ValueError("删除前路径检查失败。")
        if item["directory"]:shutil.rmtree(p)
        else:p.unlink()
        if item["logical_bytes"]>100*1024**2:print("已清理",p,round(item["logical_bytes"]/1024**3,2),"GiB（逻辑）",flush=True)
    verify_model()
    for name,digest in snapshot.items():
        if sha(Path(name))!=digest:raise ValueError("受保护文件发生变化："+name)
    after={drive:shutil.disk_usage(drive+":/").free for drive in ["C","F"]}
    result={"status":"complete","deleted_targets":len(plan),"logical_deleted_bytes":sum(x["logical_bytes"] for x in plan),
        "disk_free_before":before,"disk_free_after":after,"reclaimed_bytes":{d:after[d]-before[d] for d in before},
        "final_model":str(KEEP),"final_model_files_verified":len(manifest)+1,"protected_files_verified":len(snapshot),
        "preserved_runtime_environments":[str(CACHE/"comp5434-english/venv"),str(CACHE/"comp5434-round5/venv")],
        "training_reproduction":"historical caches/checkpoints removed by user request; inference weights/source/docs/submissions/raw data retained"}
    (LOG/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true")
    if parser.parse_args().execute:main()
    else:print("此工具删除历史训练缓存；确认保留第十轮后使用 --execute。已完成的清理不会重复执行。")
