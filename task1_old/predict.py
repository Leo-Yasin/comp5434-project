"""从保存的 JSON 模型重新生成提交文件，不需要重新训练。

示例：python -m task1.predict --model task1/outputs/baseline/forest/model.json
      --test comp-5434-20261-project-task-1/test.csv --output task1/outputs/reproduced
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import FEATURES, Preprocessor
from .models import model_from_dict
from .run import export_submission


def main():
    parser = argparse.ArgumentParser(description="加载 Task 1 模型并复现预测")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.model.read_text(encoding="utf-8"))
    if state.get("schema_version") != 1:
        raise ValueError("不支持的模型文件版本。")
    test = pd.read_csv(args.test)
    if set(test.columns) != set(FEATURES + ["id"]) or not test.id.is_unique or test.id.isna().any():
        raise ValueError("测试集字段或 id 不合法。")
    prep = Preprocessor.from_dict(state["preprocessor"])
    model = model_from_dict(state["model"])
    prediction = model.predict(prep.transform(test))
    args.output.mkdir(parents=True, exist_ok=False)
    export_submission(test, prediction, args.output)
    print(f"已复现 {len(test)} 条预测：{args.output}")


if __name__ == "__main__":
    main()
