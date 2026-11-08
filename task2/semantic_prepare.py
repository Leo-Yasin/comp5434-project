"""第五轮语义实验：固定验证文本抽取和外部模型下载。

这里只抽取输入字段，绝不下载原始评论标签或比赛测试答案。模型版本固定到提交
哈希，下载后记录 SHA-256；后续报告必须说明外部模型训练语料与本数据重叠未知。
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import ROOT, CSV_OPTIONS, sha256, write_json


MODEL_REPOSITORY = "Xenova/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "fad9b372f770c208d2efaf00d02f2cb3b61ec450"
OUT = ROOT / "task2/outputs/round5_20261105"


def download():
    """仅下载固定版本公开模型；临时文件完成后再改名，断线不产生伪完整文件。"""
    folder = OUT / "external_model"
    folder.mkdir(parents=True, exist_ok=True)
    files = ["README.md", "config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "vocab.txt", "onnx/model_fp16.onnx",
             "onnx/model_quantized.onnx"]
    records = []
    for name in files:
        destination = folder / name
        destination.parent.mkdir(exist_ok=True)
        began = time.perf_counter()
        if not destination.exists():
            url = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{name}"
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                while True:
                    block = response.read(4 * 1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
            temporary.replace(destination)
        records.append({"file": name, "bytes": destination.stat().st_size,
                        "sha256": sha256(destination), "seconds": time.perf_counter()-began})
        print("模型文件就绪", records[-1], flush=True)
        write_json(folder / "provenance.json", {
            "repository": MODEL_REPOSITORY, "revision": MODEL_REVISION,
            "original_model": "nlptown/bert-base-multilingual-uncased-sentiment",
            "original_model_card": "https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment",
            "license_original_model": "MIT", "competition_external_model_rule": "not_yet_verified",
            "external_training_overlap_with_competition": "unknown", "files": records})
    # 附原作者模型卡，保留原许可证声明与训练语料说明，不只保留第三方导出说明。
    original_revision = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"
    card = folder/"ORIGINAL_MODEL_CARD.md"
    if not card.exists():
        url = f"https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment/resolve/{original_revision}/README.md"
        card.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    provenance = json.loads((folder/"provenance.json").read_text(encoding="utf-8"))
    provenance["original_model_revision_for_card"] = original_revision
    provenance["original_model_card_sha256"] = sha256(card)
    write_json(folder/"provenance.json", provenance)


def extract():
    """按既有验证 id 抽取，文本与旧模型预测严格同序；不重新挑选容易的样本。"""
    OUT.mkdir(parents=True, exist_ok=True)
    expected = np.load(ROOT / "task2/outputs/round3_20261105/development.npz")["id"]
    collected = []
    began = time.perf_counter()
    for i, chunk in enumerate(pd.read_csv(ROOT / "comp-5434-20261-project-task-2/train.csv",
                                          chunksize=100000, usecols=["id", "title", "comment"], **CSV_OPTIONS)):
        collected.append(chunk.loc[chunk.id.isin(expected)])
        if i % 15 == 0:
            print("抽取验证文本", i, flush=True)
    frame = pd.concat(collected, ignore_index=True)
    if not np.array_equal(frame.id.to_numpy(), expected):
        raise ValueError("语义验证输入与原验证 id 不一致。")
    frame.fillna("").to_csv(OUT / "validation_text.csv", index=False)
    test = pd.read_csv(ROOT / "comp-5434-20261-project-task-2/test.csv", **CSV_OPTIONS)
    test[["id", "title", "comment"]].fillna("").to_csv(OUT / "test_text.csv", index=False)
    write_json(OUT / "extraction.json", {"validation_rows": len(frame), "test_rows": len(test),
        "seconds": time.perf_counter()-began, "input_fields": ["title", "comment"],
        "validation_id_sha256": sha256(OUT / "validation_text.csv")})


def subset():
    """按另一段组哈希抽取3/8，固定约5.8万条；独立于评分、文本质量及模型误差。"""
    data = np.load(ROOT / "task2/outputs/round3_20261105/development.npz")
    selected = ((data["groups"]//np.uint64(500)) % np.uint64(8)) < 3
    index = np.flatnonzero(selected)
    text = pd.read_csv(OUT/"validation_text.csv", keep_default_na=False)
    if not np.array_equal(text.id.to_numpy(), data["id"]):
        raise ValueError("抽样前 id 不匹配。")
    text.iloc[index].to_csv(OUT/"semantic_development_text.csv", index=False)
    np.savez_compressed(OUT/"semantic_selection.npz", index=index, id=data["id"][index])
    write_json(OUT/"semantic_selection.json", {"rule": "(groups // 500) % 8 < 3",
        "rows": len(index), "not_label_or_prediction_dependent": True,
        "chosen_before_semantic_effect_evaluation": True})
    print("固定语义开发子集", len(index), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["download", "extract", "subset"], required=True)
    args = parser.parse_args()
    {"download": download, "extract": extract, "subset": subset}[args.stage]()
