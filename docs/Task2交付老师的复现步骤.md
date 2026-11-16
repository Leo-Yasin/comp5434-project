# Task2 Round10：交付与复现步骤

本方案已经停止训练。原参考CSV的Kaggle RMSE=**0.49880**，来自用户实际测评反馈。
模型包保持原Round10的全部权重，不需要重新训练。
原Windows环境完整10000行冷启动已经完成，耗时39.18分钟，平均评分差2.08e-8、最大2.44e-6，CSV不是逐字节相同。
macOS M1上的实质差异尚未定位；全量数字及验证边界见[复现排查记录](Task2跨机复现差异排查-20261116.md)。

## 1. 准备交付内容

- 当前GitHub源码及本说明，包含Round10推理入口和中文原理文档。
- [Round10完整模型Release](https://github.com/Leo-Yasin/comp5434-project/releases/tag/task2-round10-model-20261115)，含全部父模型、分词器、统计表和最终头。
- 比赛原始`test.csv`，由课程比赛提供，不随公开Git仓库上传。
- 原参考预测`submissions/latest/task2.csv`（本地历史路径为`submissions/20261114_task2_round10/task2_round10.csv`）。
- 本次完整冷启动的验收报告及原机环境诊断JSON（GitHub的`docs/repro-20261116/`）。

原参考CSV只用于核验，预测程序仍需从原始CSV运行网络和全部组合模型。
Release内的UNDERSTANDING.md为发布时版本，最新环境及验收边界以当前仓库文档为准。

## 2. 按原基线准备三个环境

优先采用Windows x64；原机为Python3.9.13数值环境、Python3.12.14旧BERT环境和Python3.12.14英语环境。
这只是明确已验证/正在验证的配置，不保证所有Windows CPU或ARM设备的数值都相同。
三套角色的tokenizers版本不同，不要混装到一个环境。安装Python后，在仓库根目录执行：

```powershell
py -3.9 -m venv .venv-numeric
py -3.12 -m venv .venv-semantic
py -3.12 -m venv .venv-english
& .venv-numeric/Scripts/python.exe -m pip install -r task2/requirements.txt
& .venv-semantic/Scripts/python.exe -m pip install -r task2/requirements-semantic.txt
& .venv-english/Scripts/python.exe -m pip install -r task2/requirements-round10-english-win.txt
```

`py -3.9`和`py -3.12`使用电脑已安装的对应版本，请先核对实际版本。
旧BERT安装的是DirectML发行包，但原推理明确选择CPUExecutionProvider。
英语适配的最后层预测也需要torch2.6.0+cpu，不能漏装。
macOS不能直接安装Windows专用DirectML包，不应静默替换运行库后声称环境已经一致。

## 3. 下载并校验完整模型

```powershell
& .venv-numeric/Scripts/python.exe tools/download_task2_model.py --download-dir C:/COMP5434-downloads --destination C:/COMP5434-models/task2-round10
Get-FileHash -Algorithm SHA256 comp-5434-20261-project-task-2/test.csv
```

下载工具会检查分卷、ZIP及全部模型文件SHA。
原始测试CSV的SHA256应为`2c0b99a3b504da114ab6e6c41243b92668efc73449a2841ecfa3b95125168bce`。
若只改变了CSV换行，SHA也会变化；应保留原文件进行首次验收，避免在排查中同时改变输入。

## 4. 从空工作目录做一次完整预测

```powershell
& .venv-numeric/Scripts/python.exe -X utf8 tools/run_round10_checked.py --model C:/COMP5434-models/task2-round10/model --test comp-5434-20261-project-task-2/test.csv --semantic-python .venv-semantic/Scripts/python.exe --english-python .venv-english/Scripts/python.exe --work-dir C:/COMP5434-models/inference-round10-checked --output task2_prediction_round10.csv
```

首次工作目录必须为空，输出CSV不能预先存在。请保留完整10000行输入，不拆文件、不改批量或排序。
新入口先检查依赖并绑定环境；续跑时输入、源码、权重或环境不同会拒绝复用缓存。
此命令只执行预测，不下载额外预训练模型，不读取训练缓存或已有最终评分。
首次可在命令末尾加`--check-only`先检查模型SHA、依赖与环境绑定；检查通过后去掉该参数运行完整预测。

## 5. 生成验收报告

```powershell
& .venv-numeric/Scripts/python.exe -X utf8 tools/collect_round10_diagnostics.py --model C:/COMP5434-models/task2-round10/model --test comp-5434-20261-project-task-2/test.csv --semantic-python .venv-semantic/Scripts/python.exe --english-python .venv-english/Scripts/python.exe --work-dir C:/COMP5434-models/inference-round10-checked --prediction task2_prediction_round10.csv --reference submissions/latest/task2.csv --output round10_diagnostics.json
```

核对报告中的模型SHA异常列表为空、10000个唯一ID且顺序一致、所有评分有限且在1到5之间。
`prediction_comparison`给出相对参考CSV的平均/最大/RMS差及两个文件的SHA；这不是相对测试真实标签的RMSE。
完全相同与仅数值近似相同应分别记录，不能把实质差异称作浮点末位误差。
如有实质差异，保留CSV、parent.npz、工作目录和诊断报告，先逐阶段比较，不修改权重或盲目重跑。

原参考CSV的SHA256为`000952138fb52c63dd65b7bcd859a86401144e8dfb18dbdeeb69983a8891b891`。
只有原参考CSV已经实测得到0.49880；变化后的新预测是否取得相同显示分数，仍以Kaggle实际测评为准。
