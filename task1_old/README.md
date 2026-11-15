# GitHub最新快照入口（2026-11-08）

当前上传版本为第二轮最终混合模型，提交文件位于 `../submissions/latest/task1.csv`。
完整模型在 `latest/model/`，不是后文历史实验中的core_expert路径。
用户仅反馈线上仍为0.913级别，未提供最新精确分数。

在项目根目录、已安装本任务requirements的环境中执行：

```powershell
python -m task1.round2_predict --model task1/latest/model --test comp-5434-20261-project-task-1/test.csv --output task1_prediction.csv
python -m unittest discover -s task1/tests -v
```

测试数据由队友自行下载；下面保留原始开发说明和历史实验路径，outputs未随仓库上传。

---

# COMP5434 Task 1：天气分类

本目录提供从原始 CSV、训练验证、模型保存到提交文件复现的完整流程。
代码中的算法、数据流与关键边界均附中文注释。

## 自主实现内容

1. Softmax：稳定概率计算、交叉熵、L2 梯度、全批量更新及内层早停。
2. CART：Gini 分裂目标、排序累计计数、全部有效阈值搜索和递归建树。
3. 随机森林：Bootstrap 抽样、节点随机特征子集及概率平均。
4. 分层交叉验证、Macro-F1、混淆矩阵和训练折内预处理。

NumPy/pandas 用于底层运算与 CSV 处理，没有调用现成分类器。

## 环境

建议 Python 3.12。以下命令均在本项目根目录（task1 的上级目录）执行。

```powershell
python -m pip install -r task1/requirements.txt
python -m unittest discover -s task1/tests -v
```

## 完整运行

```powershell
python -m task1.run --output-dir task1/outputs/baseline
```

默认读取 `comp-5434-20261-project-task-1/`。输出目录必须不存在，避免覆盖既有实验。
省略 `--output-dir` 时以北京时间生成独立时间目录。

只运行一个模型或增加异常标记：

```powershell
python -m task1.run --models forest --output-dir task1/outputs/forest_flags --anomaly-flags
```

通过 `python -m task1.run --help` 查看深度、叶子样本数、树数量等参数。

## 验证与计时

- 默认种子 5434，分层 5 折，各模型共用相同折号。
- 预处理只拟合训练折，外层验证样本不用于均值、词表或梯度更新。
- Softmax 在外层训练折内部留出 15% 选择轮数，再从零训练整个训练折。
- 全量 Softmax 的轮数取各折内部选择轮数的中位数。
- OOF 表示每个样本都由没有训练过该样本的模型预测。
- CV 均值用于开发时比较模型；在这些模型中选择最优后，成绩不等于独立测试成绩。
- 全量离线时间为该次全量预处理 + 模型训练，不含先前交叉验证搜索及模型写盘。
- 全测试推理时间包含测试特征转换和预测，不含写 CSV。
- Softmax history 记录每轮耗时；森林 history 记录每棵树耗时，树数量不称为 epoch。
- 计时为单次实测，不能视为稳定的多次运行基准；环境与线程相关信息见 manifest。

## 保存结果与重新预测

每次运行产生：

```text
outputs/baseline/
  config.json                 所有参数
  manifest.json               环境、代码/数据哈希与运行状态
  data_audit.json              数据检查
  folds.csv                   固定折号
  comparison.json             模型比较
  results.md                  中文结果报告
  softmax/ cart/ forest/
    fold_*_metrics.json       逐折指标与耗时
    oof_predictions.csv       外层验证预测及概率
    summary.json              汇总结果
    model.json                全量预处理状态与模型
    submission_integer.csv    整数标签 1–4
    submission_weather.csv    Sunny/Cloudy/Rainy/Snowy
```

独立复现森林预测：

```powershell
python -m task1.predict --model task1/outputs/baseline/forest/model.json --test comp-5434-20261-project-task-1/test.csv --output task1/outputs/reproduced_forest
```

训练入口会自动重新加载保存的模型并检查概率与标签一致性。独立预测命令不读取训练标签。
同一软件环境、配置与输入下目标是复现相同预测；不同 BLAS/CPU 的浮点细节可能有差异。

进一步核验全部 OOF 指标和提交文件，并从零重训森林：

```powershell
python -m task1.verify --run task1/outputs/baseline --retrain-forest
python -m task1.verify --run task1/outputs/forest_flags
python -m task1.summarize
```

最后一条命令要求 baseline 和 forest_flags 均已完成，生成中文实验总结与基础森林错分明细。
当前运行已完成，默认输出目录已存在；再次训练请换一个新的目录名。

## 本机实测结果（2026-10-19）

| 模型 | 5 折 Macro-F1 均值 | 折间样本标准差 |
|---|---:|---:|
| Softmax | 0.866689 | 0.007396 |
| CART | 0.903955 | 0.005070 |
| 随机森林 | 0.915223 | 0.001679 |
| 随机森林 + 异常标记 | 0.914683 | 0.003409 |

当前保留基础森林。七项核心测试、模型加载预测检查、森林从零重训一致性检查均已通过。
独立预测命令生成的两个 CSV 与原文件逐字节相同。尚未上传 Kaggle。

## 提交格式：使用整数标签（2026-10-27 线上错误反馈后更正）

用户提交名称版后，Kaggle 返回 `The input string 'Rainy' was not in a correct format.`。
这说明评测器需要数值标签，即使提供的样例使用名称，也不能据此推荐名称版。
请上传 `outputs/baseline/forest/submission_integer.csv`，列名为 id,label，标签为 1–4。
映射为 1 Sunny、2 Cloudy、3 Rainy、4 Snowy。名称版只保留用于解释，不再推荐上传。
两份文件代表相同模型预测，修改编码不需要重新训练。当前代码不会自动上传。

## 阶段文档与文件职责

- `docs/阶段记录.md`：阶段目标、设计、结论与后续事项。
- `docs/模型设计.md`：算法公式、数据流和验证协议。
- `docs/实验总结.md`：模型比较、消融结果和错误分析。
- `data.py`：输入检查、划分与预处理。
- `models.py`：三个自主模型。
- `metrics.py`：分类指标。
- `run.py`：实验调度、记录与模型导出。
- `predict.py`：加载模型后独立预测。
- `verify.py`：从记录重算指标、核对预测、可选从零重训森林。
- `summarize.py`：生成中文实验总结和 OOF 错分明细。
- `tests/test_core.py`：关键数值正确性测试。

## 2026-11-05 优化

新增自主直方图 Newton 提升树：多类梯度/Hessian、二阶增益、分位数分箱、
带 L2 的叶子更新与采样均由本项目实现。沿用原五折，比较两种深度、两种检查轮数
及预设概率融合比例，同时保留生成式/森林调参未改善的实验。

提升树与原森林各占 50% 时，原五折 Macro-F1 从 0.915223 变为 0.916247。
冻结该配置后，第二组五折由 0.914350 变为 0.915553；这是小幅改善，尚不是新 Kaggle 成绩。
模型目录 `outputs/optimization_20261105/` 包含模型、基线依赖和独立预测证据。

```powershell
python -m task1.optimize
python -m task1.generative_experiment
python -m task1.tune_forest
python -m task1.confirm_optimization
python -m task1.predict_optimized --model task1/outputs/optimization_20261105 --test comp-5434-20261-project-task-1/test.csv --output task1/outputs/optimization_20261105/reproduced.csv
```

这些命令是首次实验的执行顺序，本机输出已存在，不应重复覆盖。统一的新上传文件和
SHA-256 清单见项目根目录 submissions/20261105/；原基线保留不变。
全部优化结果见项目根目录 docs/优化结果-20261105.md。

## 第二轮优化（2026-11-05）

继续比较混合高斯-类别 EM、随机阈值森林、类别支持专家和先验配额分配。
前三者全部自主实现；配额分配不用预测集标签，但本地收益不佳，不作为推荐方案。
支持专家训练折内估计压力最短支持区间与类别条件频率，仅为新专家选择训练记录，
不改原始数据；推理对所有类别均不支持的记录回退第一轮模型。
原五折支持专家融合为 0.916855，相比上一轮 0.916247 仍是小幅变化。

```powershell
python -m task1.round2
python -m task1.random_threshold_experiment
python -m task1.core_experiment
python -m task1.prior_experiment
python -m task1.confirm_core
python -m task1.round2_predict --model task1/outputs/core_expert_20261105/package --test comp-5434-20261-project-task-1/test.csv --output task1/outputs/core_expert_20261105/reproduced.csv
```

上面为首次实验命令，本机结果已存在；不覆盖同名输出。训练和验证候选、未改善配置都保留。
独立 package 同时保存新专家和旧概率融合依赖，无需访问其他输出目录即可预测。
详见根目录 docs/第二轮优化记录-20261105.md。
