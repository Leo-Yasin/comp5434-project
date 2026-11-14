# COMP5434 Project：Task 1 / Task 2 最新方案

团队仓库：Leo-Yasin/comp5434-project。快照整理日期：2026-11-14。
每个任务只保留一个最新提交CSV；源码为当前快照，包含最新方案所需基础模块与测试。
历史优化记录保留供队友参考，不重复上传旧提交CSV或旧模型包。

| 任务 | 当前方案 | 最新提交 | 线上状态 |
|---|---|---|---|
| Task 1 | 混合高斯/类别分布与提升树、森林概率融合，第二轮最终候选 | [task1.csv](submissions/latest/task1.csv) | 用户报告0.913级别、小幅提升，精确值未提供 |
| Task 2 | Dask自主五星噪声词分类与冻结BERT头融合，本地第七轮 | [task2.csv](submissions/latest/task2.csv) | 新版待测；第六轮已测RMSE=0.53985 |

Task 1 使用Macro-F1（越大越好），Task 2 使用RMSE（越小越好）。
Task 2本地第七轮的噪声风险估计不等于Kaggle成绩。用户转述老师允许外部预训练模型，条件是理解模型；已提供中文原理/答辩指南。

## 队友阅读入口

- [Task 1当前入口与执行说明](task1/README.md)、[模型设计](task1/docs/模型设计.md)、[最近优化结果](docs/第二轮优化结果-20261105.md)。
- [Task 2当前入口与执行说明](task2/README.md)、[BERT原理与答辩指南](task2/docs/BERT原理与本项目答辩指南.md)、[本地第七轮结果](docs/Task2本地第七轮优化结果-20261114.md)。
- [上传范围与复现边界](docs/最新上传说明.md)、[实际Kaggle记录](docs/Kaggle测评记录.md)。
- [两个提交文件的原始来源、行数和SHA-256](submissions/latest/manifest.json)。

## 包含与不包含什么

包含当前源码、依赖说明、测试、中文阶段文档、两个最新提交CSV，以及Task 1约18MiB的完整推理模型。
Task 2保留选定配置、验证结果和指纹清单，但数百MB的完整模型包需要从维护者另行获取。
原始数据、训练缓存、Python环境、课程PDF和旧提交CSV不包含在仓库中。

数据由队友自行下载，放到根目录两个 `comp-5434-20261-project-task-*` 文件夹。
克隆后可以直接查看方案或下载提交CSV；运行Task 1预测需要测试CSV，运行Task 2预测还需要完整模型包。
所有原模型和原提交文件均在维护者本地完整保留。

## 复现与协作

Task 1的模型入口是 `task1/latest/model`；Task 2模型另行获取后放在 `task2/latest/model`。
具体预测命令见各任务README开头。Task 1与Task 2原数值环境不同，Task 2语义推理另用独立ONNX环境。
旧说明里的 `outputs/...` 是当时训练产物路径，不代表这些大文件已随Git上传。
旧逐轮整理脚本需要其对应本地产物，不能在只克隆最新版仓库后直接执行所有finish脚本。

后续修改请新建分支，代码和中文阶段记录一起提交，通过Pull Request让队友查看。
首次提交是现有成果的当前快照，不虚构此前各轮的完整Git提交历史。
