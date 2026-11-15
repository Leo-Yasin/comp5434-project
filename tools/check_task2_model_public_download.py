"""不使用凭据验证Release清单和一个二进制分卷的公开下载入口。"""
import json
from pathlib import Path
import urllib.request


def main():
    folder=Path.home()/".cache/comp5434-title10/github-release"
    proof=json.loads((folder/"published_release.json").read_text(encoding="utf-8"))
    if proof["tag"]!="task2-round10-model-20261115" or not proof["all_asset_digests_verified"]:raise ValueError("Release尚未完整发布。")
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({"https":"http://127.0.0.1:7897"}))
    asset=next(a for a in proof["assets"] if a["name"]=="task2-round10-model.manifest.json")
    with opener.open(asset["url"],timeout=60) as response:remote=json.load(response)
    local=json.loads((folder/asset["name"]).read_text(encoding="utf-8"))
    if remote!=local:raise ValueError("公开清单与原始清单不一致。")
    binary=next(a for a in proof["assets"] if a["name"].endswith(".part026"))
    request=urllib.request.Request(binary["url"],headers={"Range":"bytes=0-1023"})
    with opener.open(request,timeout=60) as response:prefix=response.read(1024);status=response.status
    with (folder/binary["name"]).open("rb") as source:expected=source.read(1024)
    if prefix!=expected:raise ValueError("公开二进制下载入口内容不一致。")
    result={"anonymous":True,"manifest_matches":True,"binary_prefix_matches":True,"binary_http_status":status,
        "release_url":proof["url"],"assets":len(proof["assets"]),"parts":len(remote["parts"]),"archive_sha256":remote["archive_sha256"]}
    (folder/"public_download_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("公开下载终检通过",result,flush=True)


if __name__=="__main__":main()
