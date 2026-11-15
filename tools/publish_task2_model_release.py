"""将完整第十轮推理模型发布为GitHub Release附件，不把大权重写入Git历史。

发布前验证每个参数SHA，打包后逐成员校验；上传后核对GitHub资产SHA。
已有Git认证仅在进程内使用，凭据不打印、不落盘。原模型/训练数据不修改。
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile
import requests

MODEL=Path.home()/".cache/comp5434-title10/model"
DEST=MODEL.parent/"github-release"
REPO="Leo-Yasin/comp5434-project"
TAG="task2-round10-model-20261115"
NAME="task2-round10-model.zip"


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def save(name,value):
    (DEST/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    DEST.mkdir(exist_ok=True)
    manifest=json.loads((MODEL/"file_integrity.json").read_text(encoding="utf-8"))
    expected={**manifest,"file_integrity.json":{"bytes":(MODEL/"file_integrity.json").stat().st_size,"sha256":digest(MODEL/"file_integrity.json")}}
    for relative,item in expected.items():
        path=MODEL/relative
        if MODEL.resolve() not in path.resolve().parents:raise ValueError("模型清单路径越界。")
        if path.stat().st_size!=item["bytes"] or digest(path)!=item["sha256"]:raise ValueError("模型参数指纹变化："+relative)
    print("模型完整性通过",len(expected),"文件",sum(v["bytes"] for v in expected.values()),"字节",flush=True)
    archive=DEST/NAME
    if not archive.exists():
        partial=archive.with_suffix(".zip.partial")
        if partial.exists():raise FileExistsError("发现未完成压缩包，保留现场后再恢复。")
        with zipfile.ZipFile(partial,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as output:
            for relative in sorted(expected):
                output.write(MODEL/relative,"model/"+relative)
                if expected[relative]["bytes"]>50*1024*1024:print("已打包",relative,flush=True)
        partial.replace(archive)
    with zipfile.ZipFile(archive) as packed:
        if set(packed.namelist())!={"model/"+r for r in expected}:raise ValueError("压缩包成员不匹配。")
        for relative,item in expected.items():
            h=hashlib.sha256()
            with packed.open("model/"+relative) as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
            if h.hexdigest()!=item["sha256"]:raise ValueError("压缩包内参数校验失败："+relative)
    size=archive.stat().st_size
    if size>=2*1024**3:raise ValueError("附件超过GitHub Release单文件上限。")
    checksum=digest(archive)
    (DEST/(NAME+".sha256")).write_text(checksum+"  "+NAME+"\n",encoding="ascii")
    bundle={"schema_version":1,"round":10,"archive":NAME,"archive_bytes":size,"archive_sha256":checksum,
        "uncompressed_bytes":sum(v["bytes"] for v in expected.values()),"members":len(expected),"files":expected,
        "source_commit":"a1d2c1439eddd5603c97c405b5de70b543d3b94d","new_kaggle_score":None,
        "contents":"complete inference weights/config/tokenizers/history statistics; no raw datasets or training caches"}
    # 大请求曾出现代理写入超时，改为64MiB分卷；失败只重传一卷，完整ZIP SHA保持不变。
    part_paths=[];parts=[]
    with archive.open("rb") as source:
        number=1
        while True:
            block=source.read(64*1024*1024)
            if not block:break
            part=DEST/(NAME+f".part{number:03d}");expected_part=hashlib.sha256(block).hexdigest()
            if not part.exists():part.write_bytes(block)
            if part.stat().st_size!=len(block) or digest(part)!=expected_part:raise ValueError("本地分卷校验失败。")
            part_paths.append(part);parts.append({"name":part.name,"bytes":len(block),"sha256":expected_part});number+=1
    bundle.update(transport="split_zip",parts=parts)
    save("task2-round10-model.manifest.json",bundle)
    print("压缩包校验通过",size,"字节 SHA256",checksum,flush=True)
    # 使用GCM现有认证，绝不把密码/令牌写到日志、文件或命令参数中。
    auth=subprocess.run(["git","credential","fill"],input="protocol=https\nhost=github.com\n\n",text=True,capture_output=True,check=True)
    credential=dict(row.split("=",1) for row in auth.stdout.splitlines() if "=" in row)
    session=requests.Session();session.proxies.update({"https":"http://127.0.0.1:7897"})
    session.headers.update({"Authorization":"Bearer "+credential["password"],"Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
    base="https://api.github.com/repos/"+REPO
    def check(response):
        if not response.ok:raise RuntimeError("GitHub请求失败 HTTP "+str(response.status_code)+" "+str(response.json().get("message","")))
        return response.json()
    source="2ff6786f9a86370b3cf5b024b9041a4d84b47b98"
    ref=session.get(base+"/git/ref/tags/"+TAG,timeout=60)
    if ref.status_code==404:check(session.post(base+"/git/refs",json={"ref":"refs/tags/"+TAG,"sha":source},timeout=60))
    elif check(ref)["object"]["sha"]!=source:raise ValueError("版本标签指向其他源码，不覆盖。")
    found=session.get(base+"/releases",timeout=60);releases=check(found)
    release=next((r for r in releases if r["tag_name"]==TAG),None)
    if release is None:
        release=check(session.post(base+"/releases",json={"tag_name":TAG,"target_commitish":"2ff6786f9a86370b3cf5b024b9041a4d84b47b98",
            "name":"Task2 Round 10 — 完整模型与权重","draft":True,
            "body":"完整第十轮推理模型，含全部父轮权重、分词器、配置与统计参数。模型以64MiB ZIP分卷上传，按仓库 tools/download_task2_model.py 自动下载、合并、校验并解压，得到model文件夹。\n\n不需要原始训练数据/训练缓存；运行预测仍需自己的测试CSV与依赖环境。每卷及整包SHA-256见manifest附件。代码提交a1d2c1439eddd5603c97c405b5de70b543d3b94d，新Kaggle分数未知。GitHub自动提供的Source code(zip)仅含代码。"},timeout=60))
    release=check(session.patch(base+f"/releases/{release['id']}",json={"body":"完整第十轮推理模型（约1.99GB），含全部父轮权重、分词器、配置和统计参数。64MiB ZIP分卷，支持失败重传。请运行仓库 tools/download_task2_model.py 自动下载、合并、校验、解压；得到model目录后用--model指定。\n\n每卷、整包和每个模型文件的SHA均保存在manifest附件中。原始训练数据和临时训练缓存无需下载。GitHub自动生成的Source code(zip)仅包含代码，不包含权重。\n\n模型对应代码a1d2c1439eddd5603c97c405b5de70b543d3b94d。"},timeout=60))
    release=check(session.patch(base+f"/releases/{release['id']}",json={"tag_name":TAG,"target_commitish":source},timeout=60))
    save("release_in_progress.json",{"release_id":release["id"],"tag":TAG,"draft":release["draft"]})
    class Progress:
        def __init__(self,path):self.file=path.open("rb");self.size=path.stat().st_size;self.last=time.monotonic();self.start=self.last
        def __len__(self):return self.size
        def tell(self):return self.file.tell()
        def seek(self,*a):return self.file.seek(*a)
        def read(self,n=-1):
            data=self.file.read(n)
            if time.monotonic()-self.last>=30:
                pos=self.tell();print("上传进度",round(100*pos/self.size,1),"%",round(pos/1024**2),"MiB，",round(pos/max(time.monotonic()-self.start,1)/1024**2,2),"MiB/s",flush=True);self.last=time.monotonic()
            return data
    uploaded=[]
    for path in [DEST/(NAME+".sha256"),DEST/"task2-round10-model.manifest.json",*part_paths]:
        assets=check(session.get(base+f"/releases/{release['id']}/assets",timeout=60))
        existing=next((a for a in assets if a["name"]==path.name),None)
        local_sha=digest(path)
        if existing and existing["state"]=="uploaded" and existing["size"]==path.stat().st_size and existing.get("digest")=="sha256:"+local_sha:
            asset=existing
        else:
            if existing:
                # 只移除本工具此专属Release同名的不完整附件，不触碰其他资产。
                response=session.delete(base+f"/releases/assets/{existing['id']}",timeout=60)
                if response.status_code!=204:raise RuntimeError("无法移除未完成的同名附件。")
            for attempt in range(1,4):
                reader=Progress(path)
                try:
                    print("开始上传",path.name,path.stat().st_size,"字节，第",attempt,"次",flush=True)
                    response=session.post(release["upload_url"].split("{")[0],params={"name":path.name},data=reader,
                        headers={"Content-Type":"application/octet-stream","Content-Length":str(path.stat().st_size)},timeout=(180,3600))
                    asset=check(response);break
                except (requests.RequestException,RuntimeError):
                    if attempt==3:raise
                    print("本卷上传失败，重试",path.name,flush=True)
                    stale=check(session.get(base+f"/releases/{release['id']}/assets",timeout=60))
                    for item in stale:
                        if item["name"]==path.name:
                            session.delete(base+f"/releases/assets/{item['id']}",timeout=60)
                finally:reader.file.close()
        if asset["size"]!=path.stat().st_size or asset.get("digest")!="sha256:"+local_sha:raise RuntimeError("GitHub附件大小/SHA验证失败。")
        uploaded.append({"name":path.name,"bytes":asset["size"],"sha256":local_sha,"url":asset["browser_download_url"]})
        print("远端附件SHA通过",path.name,flush=True)
    release=check(session.patch(base+f"/releases/{release['id']}",json={"draft":False,"tag_name":TAG,"target_commitish":source},timeout=60))
    save("published_release.json",{"url":release["html_url"],"tag":TAG,"assets":uploaded,"model_manifest_verified":True})
    print("完整模型发布完成",release["html_url"],flush=True)


if __name__=="__main__":main()
