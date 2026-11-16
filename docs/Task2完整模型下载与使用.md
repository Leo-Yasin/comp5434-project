# Task2第十轮完整模型：下载与使用

完整模型附件位于[GitHub Release：Task2 Round 10](https://github.com/Leo-Yasin/comp5434-project/releases/tag/task2-round10-model-20261115)。
它包含66个文件、约1.99GB，覆盖最终标题残差头、第九轮适配权重、全部父轮模型、分词器、配置及推理所需统计参数。
普通`git clone`或GitHub的`Source code (zip)`只下载源码，不会自动带上Release里的大型权重。

## 队友推荐操作

先更新代码，然后在仓库根目录执行以下命令。下载与解压建议在C盘短路径预留6GB以上空间。
脚本仅使用Python标准库，不必先安装模型运行依赖。

```powershell
git pull
python tools/download_task2_model.py --download-dir C:/COMP5434-downloads --destination C:/COMP5434-models/task2-round10
```

网络需要代理时，在命令末尾增加`--proxy http://127.0.0.1:7897`，端口按自己的实际设置修改。
工具会自动下载26个分卷，逐卷核对SHA、按顺序合并ZIP、核对整包SHA，然后解压并验证全部66个模型文件。
连接中断时重新执行相同命令即可，已经校验正确的卷不会重复下载；已有model目录不会被覆盖。

完成后的模型路径是：

```text
C:/COMP5434-models/task2-round10/model
```

也可以在Release页面手动下载全部`.part001`至`.part026`及manifest，放在同一下载目录后执行脚本。
全部文件已下载时可在命令末尾增加`--offline`，无需联网即可合并、校验和解压。
所有卷共同组成一个ZIP，单个分卷不能直接解压。
整包SHA-256为`d7e593daa5e975cbfc3e75c8e5b941ef48f94b2bf776c7e3ea4f6bc445fb945f`。

## 执行预测

2026-11-16状态：用户报告原参考CSV的Kaggle RMSE=0.49880，但朋友在macOS M1的冷启动输出存在实质差异。
跨机复现正在按[排查记录](Task2跨机复现差异排查-20261116.md)验证，不能把原CSV分数自动标到其他机器的新输出上。
原机器使用Windows x64和三套独立环境。第九/十轮英语推理需要torch，
Windows x64英语依赖请使用`task2/requirements-round10-english-win.txt`；不要把三套互不兼容的tokenizers版本混装。

按`task2/docs/第十轮标题语义与复现.md`准备数值、旧BERT及英语CPU三个环境，然后执行：

```powershell
python tools/run_round10_checked.py --model C:/COMP5434-models/task2-round10/model --test comp-5434-20261-project-task-2/test.csv --semantic-python .venv-semantic/Scripts/python.exe --english-python .venv-english/Scripts/python.exe --work-dir C:/COMP5434-models/inference-round10-checked --output task2_prediction_round10.csv
```

命令开头的python应来自数值环境，其余两个Python路径替换成自己的实际环境路径。
预测可以使用CPU，不需要GPU，也不需要原始训练集或训练缓存；测试CSV需自行准备。
本入口首次需要空工作目录，续跑会检查环境、输入、源码与模型指纹。
必须把完整10000行测试文件一次交给入口；不要另行拆分、改变批量或排序，旧量化BERT对拼批组成敏感。
模型理解、评分噪声假设、验证边界和实际结果见第十轮对应文档。新Kaggle分数仍以用户实际测评为准。

## 维护者本机位置

当前已经验证的完整模型仍保存在`C:/Users/Leo/.cache/comp5434-title10/model`，维护者本机无需重复下载。
本地压缩包及分卷已在2026-11-15磁盘清理中删除，GitHub Release附件保留；完整model目录和66文件指纹未改动。
