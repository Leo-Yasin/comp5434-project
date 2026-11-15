"""仅在完整模型Release已公开且附件SHA全部通过后更新仓库下载入口。"""
import json
from pathlib import Path
import shutil
import task2
from task2.audit import ROOT,write_json


def main():
    folder=Path.home()/".cache/comp5434-title10/github-release"
    proof=json.loads((folder/"published_release.json").read_text(encoding="utf-8"))
    manifest=json.loads((folder/"task2-round10-model.manifest.json").read_text(encoding="utf-8"))
    public=json.loads((folder/"public_download_check.json").read_text(encoding="utf-8"))
    if not proof["all_asset_digests_verified"] or not public["manifest_matches"] or not public["binary_prefix_matches"]:raise RuntimeError("尚未完成Release/公开下载终检。")
    url=proof["url"];base="https://github.com/Leo-Yasin/comp5434-project/releases/download/"+proof["tag"]+"/"
    record={"release_url":url,"tag":proof["tag"],"manifest_url":base+"task2-round10-model.manifest.json",
        "archive_name":manifest["archive"],"archive_bytes":manifest["archive_bytes"],"archive_sha256":manifest["archive_sha256"],
        "uncompressed_bytes":manifest["uncompressed_bytes"],"parts":len(manifest["parts"]),"model_files":manifest["members"],
        "models_on_github_releases":True,"parameters_in_git":False,"all_asset_digests_verified":True,"anonymous_download_verified":True,
        "source_commit":manifest["source_commit"],"tag_commit":"2ff6786f9a86370b3cf5b024b9041a4d84b47b98"}
    pub=ROOT/"tmp/github-publish-20261108"
    header="## 完整模型已上传GitHub\n\n[下载完整模型Release]("+url+")，或按[自动下载与使用说明](docs/Task2完整模型下载与使用.md)操作。"
    header+="模型约1.99GB，以26个分卷分发；工具自动下载、合并、校验全部66文件。普通git clone不包含Release附件。\n\n"
    top=pub/"README.md";text=top.read_text(encoding="utf-8")
    text=text.replace("Task 2保留选定配置、验证结果和指纹清单，但本轮约2GB的完整模型包需要从维护者另行获取。","Task 2完整模型现已上传GitHub Release；普通Git保留源码与指纹，权重通过Release附件下载。")
    text=text.replace("## 队友阅读入口",header+"## 队友阅读入口",1);top.write_text(text,encoding="utf-8")
    task_header="# 完整模型下载入口\n\n[GitHub Release]("+url+")。详见根目录docs/Task2完整模型下载与使用.md。\n\n"
    task_header+="```powershell\npython tools/download_task2_model.py --download-dir C:/COMP5434-downloads --destination C:/COMP5434-models/task2-round10\n```\n\n解压后模型为C:/COMP5434-models/task2-round10/model，在预测命令的--model中指定即可。下载及解压建议预留6GB空间。\n\n---\n\n"
    for base_root in [ROOT,pub]:
        path=base_root/"task2/README.md";path.write_text(task_header+path.read_text(encoding="utf-8"),encoding="utf-8")
    write_json(pub/"task2/latest/model_release.json",record)
    ref=pub/"task2/latest/model_reference.json";metadata=json.loads(ref.read_text(encoding="utf-8"));metadata.update(release_url=url,manifest_url=record["manifest_url"],models_on_github_releases=True);write_json(ref,metadata)
    latest=pub/"task2/latest/README.md";latest.write_text("完整模型现可从[GitHub Release]("+url+")下载，自动下载脚本和C盘示例见根目录docs/Task2完整模型下载与使用.md。\n\n"+latest.read_text(encoding="utf-8"),encoding="utf-8")
    for relative in ["docs/Task2完整模型下载与使用.md","docs/Task2完整模型上传记录-20261115.md","tools/download_task2_model.py",
                     "tools/publish_task2_model_release.py","tools/publish_task2_model_parts.py","tools/link_task2_model_release.py"]:
        target=pub/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
    shutil.copy2(ROOT/"tools/check_task2_model_public_download.py",pub/"tools/check_task2_model_public_download.py")
    submissions=pub/"submissions/latest/manifest.json";metadata=json.loads(submissions.read_text(encoding="utf-8"));metadata["tasks"]["task2"].update(model_release_url=url,model_manifest_url=record["manifest_url"],model_archive_sha256=record["archive_sha256"]);write_json(submissions,metadata)
    upload=pub/"docs/最新上传说明.md";upload.write_text("# 2026-11-15：Task2完整权重已上传\n\n[完整模型Release]("+url+")现含所有推理参数，见自动下载指南。无需再私下另取。模型未写入Git历史。\n\n---\n\n"+upload.read_text(encoding="utf-8"),encoding="utf-8")
    root_meta=ROOT/"task2/outputs/round10_20261114/model_release.json";write_json(root_meta,record)
    print("模型下载入口已写入仓库",url,flush=True)


if __name__=="__main__":main()
