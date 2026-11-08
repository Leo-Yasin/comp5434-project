"""分块构建可复用的稀疏文本特征，并通过 Dask 进程执行。

HashingVectorizer 只承担通用文本转换，不是分类器或评分模型。
评分损失、目标统计、共轭梯度与并行归约在本项目中自行实现。
哈希特征无需在全量文本上学习词表，因此可统一转换训练与测试且不查看标签。
"""

from collections import OrderedDict
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from .audit import CSV_OPTIONS, group_hash, normalize, write_json


TITLE_DIM = 2 ** 15
BODY_DIM = 2 ** 18
NUMERIC_DIM = 8
TEXT_DIM = TITLE_DIM + BODY_DIM + NUMERIC_DIM
META_DIM = 9
DIM = TEXT_DIM + META_DIM
ID_FIELDS = ["user_id", "prod_id", "parent_prod_id"]
META_NAMES = ["user_mean_centered", "user_log_count", "user_cold",
              "product_mean_centered", "product_log_count", "product_cold",
              "parent_mean_centered", "parent_log_count", "parent_cold"]
_CACHE = OrderedDict()


def build_text(frame):
    """标题、正文分别哈希，保留否定词，加入一元/二元词组。

    标题和正文各自 L2 归一化；标题权重为 1.5。缺失文本替换为空串。
    数值变换使用预先固定的尺度，不利用验证数据估计参数。
    """
    frame = normalize(frame)
    common = dict(alternate_sign=True, norm="l2", ngram_range=(1, 2),
                  lowercase=True, strip_accents="unicode", token_pattern=r"(?u)\b\w+\b",
                  dtype=np.float32)
    title = HashingVectorizer(n_features=TITLE_DIM, **common).transform(frame.title)
    title *= 1.5
    body = HashingVectorizer(n_features=BODY_DIM, **common).transform(frame.comment)
    votes = pd.to_numeric(frame.votes, errors="coerce").fillna(0).clip(lower=0).to_numpy()
    timestamp = pd.to_numeric(frame.time, errors="coerce").fillna(0).to_numpy()
    comment_length = frame.comment.str.len().to_numpy()
    title_length = frame.title.str.len().to_numpy()
    numeric = np.column_stack([
        np.ones(len(frame)), np.log1p(votes).clip(0, 12) / 10,
        frame.purchased.str.lower().eq("true").to_numpy(dtype=float),
        np.log1p(comment_length) / 10, np.log1p(title_length) / 8,
        ((timestamp / (365.25 * 86400 * 1000) + 1970 - 2015) / 10).clip(-3, 3),
        np.log1p(frame.comment.str.count("!").to_numpy()) / 5,
        (comment_length == 0).astype(float),
    ]).astype(np.float32)
    return sparse.hstack([title, body, sparse.csr_matrix(numeric)], format="csr", dtype=np.float32)


def encode_chunk(frame, start, out, number):
    """Dask 工作进程的单个任务：编码、写缓存，只返回小型结果清单。"""
    began = time.perf_counter()
    frame = normalize(frame)
    x = build_text(frame)
    prefix = Path(out) / f"part_{number:04d}"
    sparse.save_npz(str(prefix) + ".text.npz", x, compressed=True)
    metadata = {key: pd.util.hash_pandas_object(frame[key], index=False).to_numpy(dtype=np.uint64)
                for key in ID_FIELDS}
    metadata.update(groups=group_hash(frame), rating=frame.rating.to_numpy(dtype=np.float32),
                    id=frame.id.to_numpy(dtype=np.int64))
    np.savez_compressed(str(prefix) + ".meta.npz", **metadata)
    return {"number": number, "start": start, "rows": len(frame), "nnz": int(x.nnz),
            "prefix": str(prefix.resolve()), "worker_pid": os.getpid(),
            "encode_seconds": time.perf_counter() - began}


def build_cache(client, data_dir, out, chunk_size=50000, workers=3):
    """主进程只顺序解析 CSV；并行任务队列有界，避免大量文本堆积。

    CSV 评论含换行，不能直接按任意字节切块。由 pandas 流式解析合法记录后，
    将有限数量的块提交给 Dask，比直接按字节读取更稳妥。
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pending, records, start = [], [], 0
    for number, chunk in enumerate(pd.read_csv(Path(data_dir) / "train.csv", chunksize=chunk_size, **CSV_OPTIONS)):
        # 先散发数据，再把 Future 放进任务图，避免把大型 DataFrame 嵌入图中。
        scattered = client.scatter(chunk, hash=False)
        pending.append(client.submit(encode_chunk, scattered, start, str(out), number, pure=False))
        start += len(chunk)
        if len(pending) >= workers * 2:
            record = pending.pop(0).result()
            records.append(record)
            if record["number"] % 10 == 0:
                print(f"特征缓存：完成 {record['number']+1} 块，约 {(record['start']+record['rows']):,} 条", flush=True)
    records.extend(future.result() for future in pending)
    records.sort(key=lambda r: r["number"])
    test = normalize(pd.read_csv(Path(data_dir) / "test.csv", **CSV_OPTIONS))
    test_x = build_text(test)
    sparse.save_npz(out / "test.text.npz", test_x)
    np.savez_compressed(out / "test.meta.npz", id=test.id.to_numpy(), **{
        key: pd.util.hash_pandas_object(test[key], index=False).to_numpy(dtype=np.uint64)
        for key in ID_FIELDS})
    manifest = {"rows": start, "parts": records, "test_rows": len(test),
                "text_dim": TEXT_DIM, "meta_dim": META_DIM,
                "chunk_size": chunk_size, "workers": workers,
                "worker_pids": sorted(set(record["worker_pid"] for record in records)),
                "preprocessing_seconds": time.perf_counter() - started,
                "vectorizer": {"title_dim": TITLE_DIM, "body_dim": BODY_DIM,
                               "title_weight": 1.5, "ngrams": [1, 2], "alternate_sign": True}}
    write_json(out / "manifest.json", manifest)
    return manifest


def load_partition(prefix, mode, variant):
    """每个工作进程保留有限的 LRU 缓存，避免全量数据常驻内存。

    mode=split 为开发划分，mode=full 为全量训练；variant 控制消融使用的列。
    """
    key = (prefix, mode, variant)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    data = np.load(prefix + ".meta.npz")
    y = data["rating"].astype(np.float64)
    validation = data["groups"] % np.uint64(100) < 2
    fit_mask = ~validation if mode == "split" else np.ones(len(y), dtype=bool)
    meta = np.load(prefix + f".{mode}.numeric.npy")
    if variant == "metadata":
        # 元数据消融仅用统计特征与截距，避免文本信息混入比较。
        x = sparse.csr_matrix(np.column_stack([np.ones(len(y)), meta]), dtype=np.float32)
    else:
        if Path(prefix + ".data.npy").exists():
            # mmap 只映射所需页，避免每个训练遍历都重新解压整个 CSR。
            x = sparse.csr_matrix((np.load(prefix + ".data.npy", mmap_mode="r"),
                                   np.load(prefix + ".indices.npy", mmap_mode="r"),
                                   np.load(prefix + ".indptr.npy", mmap_mode="r")),
                                  shape=(len(y), TEXT_DIM), copy=False)
        else:
            x = sparse.load_npz(prefix + ".text.npz")
        if variant == "hybrid":
            x = sparse.hstack([x, sparse.csr_matrix(meta)], format="csr", dtype=np.float32)
    result = (x, y, fit_mask, validation, data["id"])
    _CACHE[key] = result
    while len(_CACHE) > 2:
        _CACHE.popitem(last=False)
    return result
