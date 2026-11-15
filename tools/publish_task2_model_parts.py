"""并行上传已经校验的模型分卷，每卷成功后校验GitHub SHA；全部完成才发布。"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time
import requests


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    folder=Path.home()/".cache/comp5434-title10/github-release"
    manifest=json.loads((folder/"task2-round10-model.manifest.json").read_text(encoding="utf-8"))
    rid=json.loads((folder/"release_in_progress.json").read_text(encoding="utf-8"))["release_id"]
    c=dict(row.split("=",1) for row in subprocess.run(["git","credential","fill"],input="protocol=https\nhost=github.com\n\n",capture_output=True,text=True,check=True).stdout.splitlines() if "=" in row)
    def session():
        s=requests.Session();s.proxies.update({"https":"http://127.0.0.1:7897"})
        s.headers.update({"Authorization":"Bearer "+c["password"],"Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"});return s
    api="https://api.github.com/repos/Leo-Yasin/comp5434-project"
    def checked(response):
        if not response.ok:raise RuntimeError("GitHub HTTP "+str(response.status_code))
        return response.json()
    controller=session();release=checked(controller.get(api+f"/releases/{rid}",timeout=60))
    tag="task2-round10-model-20261115";source="2ff6786f9a86370b3cf5b024b9041a4d84b47b98"
    if release["tag_name"]!=tag:
        if not release["draft"] or not release["tag_name"].startswith("untagged-") or release["name"]!="Task2 Round 10 — 完整模型与权重":raise ValueError("Release不属于本次模型。")
        # GitHub草稿可能使用untagged占位；显式建立固定源码标签，再关联本工具草稿。
        ref=controller.get(api+"/git/ref/tags/"+tag,timeout=60)
        if ref.status_code==404:checked(controller.post(api+"/git/refs",json={"ref":"refs/tags/"+tag,"sha":source},timeout=60))
        elif checked(ref)["object"]["sha"]!=source:raise ValueError("既有标签指向不同源码，不覆盖。")
        release=checked(controller.patch(api+f"/releases/{rid}",json={"tag_name":tag,"target_commitish":source},timeout=60))
    items=[{"name":"task2-round10-model.zip.sha256"},{"name":"task2-round10-model.manifest.json"},*manifest["parts"]]
    class Progress:
        def __init__(self,path):self.f=path.open("rb");self.size=path.stat().st_size;self.name=path.name;self.last=time.monotonic()
        def __len__(self):return self.size
        def tell(self):return self.f.tell()
        def seek(self,*a):return self.f.seek(*a)
        def read(self,n=-1):
            block=self.f.read(n)
            if time.monotonic()-self.last>45:print(self.name,round(100*self.tell()/self.size,1),"%",flush=True);self.last=time.monotonic()
            return block
    def upload(item):
        client=session();path=folder/item["name"];size=path.stat().st_size;checksum=sha(path)
        if "sha256" in item and (checksum!=item["sha256"] or size!=item["bytes"]):raise ValueError("本地分卷变化。")
        for attempt in range(1,4):
            assets=checked(client.get(api+f"/releases/{rid}/assets",timeout=60))
            existing=next((a for a in assets if a["name"]==path.name),None)
            if existing and existing["state"]=="uploaded" and existing["size"]==size and existing.get("digest")=="sha256:"+checksum:
                print("远端SHA已匹配",path.name,flush=True);return {"name":path.name,"bytes":size,"sha256":checksum,"url":existing["browser_download_url"]}
            if existing:
                response=client.delete(api+f"/releases/assets/{existing['id']}",timeout=60)
                if response.status_code!=204:raise RuntimeError("无法清理本工具同名未完成附件。")
            body=Progress(path)
            try:
                print("上传",path.name,"尝试",attempt,flush=True)
                asset=checked(client.post(release["upload_url"].split("{")[0],params={"name":path.name},data=body,
                    headers={"Content-Type":"application/octet-stream","Content-Length":str(size)},timeout=(180,3600)))
                if asset["size"]!=size or asset.get("digest")!="sha256:"+checksum:raise RuntimeError("远端SHA不匹配。")
                print("远端SHA通过",path.name,flush=True)
                return {"name":path.name,"bytes":size,"sha256":checksum,"url":asset["browser_download_url"]}
            except (requests.RequestException,RuntimeError):
                if attempt==3:raise
                print("重试当前卷",path.name,flush=True)
            finally:body.f.close()
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(upload,item) for item in items]
        for future in as_completed(futures):
            results.append(future.result())
            (folder/"upload_progress.json").write_text(json.dumps({"verified":len(results),"total":len(items),"assets":results},ensure_ascii=False,indent=2),encoding="utf-8")
    # 所有附件逐个核对后才解除草稿状态，队友不会下载到不完整模型。
    final=checked(controller.get(api+f"/releases/{rid}/assets",timeout=60))
    for item in results:
        asset=next(a for a in final if a["name"]==item["name"])
        if asset["size"]!=item["bytes"] or asset.get("digest")!="sha256:"+item["sha256"]:raise ValueError("发布前附件终检失败。")
    published=checked(controller.patch(api+f"/releases/{rid}",json={"draft":False,"tag_name":tag,"target_commitish":source},timeout=60))
    refreshed=checked(controller.get(api+f"/releases/{rid}/assets",timeout=60))
    for item in results:item["url"]=next(a["browser_download_url"] for a in refreshed if a["name"]==item["name"])
    proof={"url":published["html_url"],"tag":published["tag_name"],"assets":sorted(results,key=lambda x:x["name"]),"model_manifest_verified":True,"all_asset_digests_verified":True}
    (folder/"published_release.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding="utf-8")
    print("模型Release发布完成",published["html_url"],flush=True)


if __name__=="__main__":main()
