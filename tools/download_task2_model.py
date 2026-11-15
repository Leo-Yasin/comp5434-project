"""自动下载GitHub Release模型分卷，验证整包及每个权重后解压。

仅使用Python标准库；网络失败重试当前卷，已通过SHA的卷无需重下。
不覆盖已有model目录，不执行模型文件中的代码，不包含GitHub凭据。
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request
import zipfile

BASE="https://github.com/Leo-Yasin/comp5434-project/releases/download/task2-round10-model-20261115/"
MANIFEST="task2-round10-model.manifest.json"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def safe_name(value):
    if not value or Path(value).name!=value or "/" in value or "\\" in value or value in [".",".."]:raise ValueError("不安全的附件名称。")
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir",type=Path,default=Path.home()/"Downloads/comp5434-task2-model")
    parser.add_argument("--destination",type=Path,default=Path("task2/latest"))
    parser.add_argument("--proxy",help="可选，例如 http://127.0.0.1:7897；不指定则使用系统/环境代理。")
    parser.add_argument("--verify-only",action="store_true",help="只验证本地分卷和整包，不下载、不解压。")
    parser.add_argument("--offline",action="store_true",help="使用已下载清单/分卷，离线合并与解压。")
    parser.add_argument("--check-only",action="store_true",help="只读取远端清单并显示大小，不下载权重。")
    args=parser.parse_args();args.download_dir.mkdir(parents=True,exist_ok=True)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({"http":args.proxy,"https":args.proxy})) if args.proxy else urllib.request.build_opener()
    if args.verify_only or args.offline:manifest=json.loads((args.download_dir/MANIFEST).read_text(encoding="utf-8"))
    else:
        with opener.open(BASE+MANIFEST,timeout=60) as response:manifest=json.load(response)
        (args.download_dir/MANIFEST).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    for part in manifest["parts"]:safe_name(part["name"])
    safe_name(manifest["archive"])
    print("模型文件",manifest["members"],"个；分卷",len(manifest["parts"]),"个；压缩",round(manifest["archive_bytes"]/1e9,2),"GB；解压",round(manifest["uncompressed_bytes"]/1e9,2),"GB",flush=True)
    if args.check_only:return
    if not args.verify_only:
        if (args.destination/"model").exists():raise FileExistsError("目标model已存在；请指定新的--destination，避免覆盖。")
        # 分卷、合并ZIP及解压模型同时保留，建议6GB以上可用空间。
        missing=sum(part["bytes"] for part in manifest["parts"] if not (args.download_dir/part["name"]).exists())
        missing+=manifest["archive_bytes"] if not (args.download_dir/manifest["archive"]).exists() else 0
        if shutil.disk_usage(args.download_dir).free<missing:raise OSError("下载盘空间不足，首次建议选择至少6GB可用的磁盘。")
    combined=hashlib.sha256();total=0
    for number,part in enumerate(manifest["parts"],1):
        path=args.download_dir/part["name"]
        if not path.exists():
            if args.verify_only or args.offline:raise FileNotFoundError(path)
            temp=path.with_suffix(path.suffix+".partial")
            for attempt in range(1,4):
                try:
                    print("下载",number,"/",len(manifest["parts"]),part["name"],flush=True)
                    with opener.open(BASE+urllib.parse.quote(part["name"]),timeout=180) as response,temp.open("wb") as output:shutil.copyfileobj(response,output,4*1024*1024)
                    if temp.stat().st_size!=part["bytes"] or sha(temp)!=part["sha256"]:raise ValueError("分卷SHA不匹配。")
                    temp.replace(path);break
                except Exception:
                    if attempt==3:raise
                    time.sleep(1)
        if path.stat().st_size!=part["bytes"] or sha(path)!=part["sha256"]:raise ValueError("已有分卷损坏："+str(path))
        with path.open("rb") as source:
            for block in iter(lambda:source.read(4*1024*1024),b""):combined.update(block);total+=len(block)
    if total!=manifest["archive_bytes"] or combined.hexdigest()!=manifest["archive_sha256"]:raise ValueError("合并后的整包SHA不匹配。")
    print("全部分卷及整包SHA通过。",flush=True)
    if args.verify_only:return
    archive=args.download_dir/manifest["archive"]
    if archive.exists():
        if sha(archive)!=manifest["archive_sha256"]:raise ValueError("已有ZIP校验失败，保留现场不覆盖。")
    else:
        temp=archive.with_suffix(".zip.partial")
        with temp.open("wb") as output:
            for part in manifest["parts"]:
                with (args.download_dir/part["name"]).open("rb") as source:shutil.copyfileobj(source,output,4*1024*1024)
        if sha(temp)!=manifest["archive_sha256"]:raise ValueError("ZIP合并校验失败。")
        temp.replace(archive)
    args.destination.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.destination).free<manifest["uncompressed_bytes"]:raise OSError("解压盘空间不足。")
    with zipfile.ZipFile(archive) as packed:
        expected={"model/"+name for name in manifest["files"]}
        if set(packed.namelist())!=expected:raise ValueError("ZIP成员清单不匹配。")
        for member in packed.infolist():
            target=(args.destination/member.filename).resolve()
            if (args.destination/"model").resolve() not in target.parents or ".." in member.filename.split("/") or "\\" in member.filename or (member.external_attr>>16)&0o170000==0o120000:raise ValueError("拒绝越界路径/符号链接。")
        packed.extractall(args.destination)
    for name,record in manifest["files"].items():
        path=args.destination/"model"/name
        if path.stat().st_size!=record["bytes"] or sha(path)!=record["sha256"]:raise ValueError("解压模型校验失败："+name)
    print("模型下载、合并、逐文件校验完成：",(args.destination/"model").resolve(),flush=True)


if __name__=="__main__":main()
