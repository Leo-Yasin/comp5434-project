"""从PyTorch官方站恢复大wheel下载，最终必须匹配官方索引SHA256。

只读已中断pip临时文件并复制到本轮C盘缓存，不删除或改动任何旧环境。
Range响应必须逐段核对Content-Range，防止代理返回其他范围导致静默损坏。
"""
import argparse
import concurrent.futures
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import time
from urllib.parse import unquote
import requests

NAME="torch-2.6.0+cu124-cp312-cp312-win_amd64.whl"
URL="https://download.pytorch.org/whl/cu124/"+NAME.replace("+","%2B")
SIZE=2532302369


def main(proxy):
    root=Path.home()/".cache/comp5434-local9/wheels";root.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.proxies={"https":proxy}
    page=session.get("https://download.pytorch.org/whl/cu124/torch/",timeout=30);page.raise_for_status()
    links=[html.unescape(link) for link in re.findall(r'href="([^"]+)"',page.text) if NAME in unquote(link)]
    hashes={link.split("#sha256=")[1] for link in links if "#sha256=" in link}
    if len(hashes)!=1:raise ValueError("官方索引未给出唯一SHA。")
    expected=hashes.pop();target=root/NAME;state=root/"ranges.json"
    if state.exists() and target.exists():record=json.loads(state.read_text(encoding="utf-8"))
    else:
        partials=list((Path.home()/"AppData/Local/Temp").glob("pip-unpack-*/"+NAME))
        first=0
        if partials:
            partial=max(partials,key=lambda p:p.stat().st_size);first=partial.stat().st_size
            with partial.open("rb") as stream:header=stream.read(2)
            if first>SIZE or (first and header!=b"PK"):raise ValueError("临时文件不是预期wheel前缀。")
            shutil.copyfile(partial,target)
        else:target.touch()
        with target.open("r+b") as stream:stream.truncate(SIZE)
        record={"prefix_bytes":first,"completed":[],"sha256":expected,"url":URL,"size":SIZE}
    if record["sha256"]!=expected or record["size"]!=SIZE:raise ValueError("下载源清单改变。")
    def save():state.write_text(json.dumps(record,indent=2),encoding="utf-8")
    save();began=time.monotonic();block=32*1024*1024
    ranges=[(first,min(SIZE-1,first+block-1)) for first in range(record["prefix_bytes"],SIZE,block)]
    def part(span):
        start,end=span
        for attempt in range(3):
            try:
                with session.get(URL+f"?comp5434_part={start}&attempt={attempt}",headers={"Range":f"bytes={start}-{end}"},
                                 stream=True,timeout=(20,60)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {start}-{end}/{SIZE}":
                        raise ValueError("服务器范围响应不匹配。")
                    done=0
                    with target.open("r+b") as stream:
                        stream.seek(start)
                        for chunk in response.iter_content(1024*1024):
                            if done+len(chunk)>end-start+1:raise ValueError("片段越界。")
                            stream.write(chunk);done+=len(chunk)
                    if done!=end-start+1:raise ValueError("片段不完整。")
                return start
            except Exception:
                if attempt==2:raise
        raise AssertionError()
    pending=[span for span in ranges if span[0] not in record["completed"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(part,span) for span in pending]):
            record["completed"].append(future.result());save()
            print("官方CUDA补齐片段",len(record["completed"]),"/",len(ranges),"秒",round(time.monotonic()-began,1),flush=True)
    digest=hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):digest.update(chunk)
    if digest.hexdigest()!=expected:raise ValueError("官方SHA不匹配，拒绝安装。")
    print("CUDA wheel官方SHA核验通过",expected,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--proxy",required=True);main(p.parse_args().proxy)
