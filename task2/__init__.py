"""COMP5434 Task 2：分块评论评分预测与 Dask 并行自定义训练。"""

# 本机旧 Anaconda 缺少 six，其他数值与 Dask 依赖均已存在。
# 仅在该依赖确实缺失时使用项目内的纯 Python 副本，不修改系统安装。
# 正常按 requirements.txt 安装的环境会直接使用环境自身的 six。
import importlib.util
import os
from pathlib import Path
import sys

# 多个 Dask 进程已提供并行，底层 BLAS 各使用一线程，避免嵌套抢占 CPU。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

if importlib.util.find_spec("six") is None:
    local_dependencies = Path(__file__).parent / ".local_deps"
    if (local_dependencies / "six.py").exists():
        sys.path.insert(0, str(local_dependencies.resolve()))
