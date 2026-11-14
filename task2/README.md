# Task2当前入口：第八轮（2026-11-14）

最新提交：submissions/latest/task2.csv。第七轮已测0.53719，本轮线上待测；本地风险不是Kaggle分数。

方案、实际选择和消融见根目录docs/Task2第八轮优化结果-20261114.md；中文复现见docs/第八轮复现说明.md，理解材料见docs/DeBERTa与量化原理说明.md。

Git不含大型权重。向维护者取得本轮整个model文件夹，放到task2/latest/model。

```powershell
python -m task2.round8_predict --model task2/latest/model --test comp-5434-20261-project-task-2/test.csv --semantic-python .venv-semantic/Scripts/python.exe --english-python .venv-english/Scripts/python.exe --work-dir task2_inference_round8 --output task2_prediction_round8.csv
python -m unittest discover -s task2/tests -v
```

新增英语环境使用Python3.12和requirements-english.txt；数值和旧BERT环境仍独立。

下方为旧阶段说明，当前入口以上文为准。

---

# Task2当前入口：本地第七轮（2026-11-14）

当前候选：Dask自主五星噪声词分类与冻结BERT头融合。最新CSV为submissions/latest/task2.csv。
此前第六轮线上0.53985，新候选线上未测评。37项测试通过，结果见根目录本地第七轮报告。

完整模型包需维护者提供，放在task2/latest/model；Git不含大型参数、原始数据或训练缓存。

```powershell
python -m task2.local7_predict --model task2/latest/model --test comp-5434-20261-project-task-2/test.csv --semantic-python .venv-semantic/Scripts/python.exe --work-dir task2_inference_local7 --output task2_prediction_local7.csv
python -m unittest discover -s task2/tests -v
```

原理与复现见docs/本地第七轮原理与复现.md。下方为旧阶段说明，当前入口以上文为准。

---

# GitHub最新入口：Task2第六轮（2026-11-08）

当前候选为冻结BERT的768维CLS+自主噪声线性锚点+64维MLP残差头，最新CSV为 `../submissions/latest/task2.csv`。
新线上分数待反馈；此前第五轮已测RMSE=0.54604。老师允许外部预训练模型（用户转述），前提是理解模型。
理解材料见 `docs/BERT原理与本项目答辩指南.md`，具体结果见根目录第六轮报告。

本仓库只包含源码、配置和验证证据，完整模型包仍需维护者单独提供，解压为 `task2/latest/model`。
原数值Python按requirements.txt安装；独立Python3.12语义环境按requirements-embedding.txt安装。
取得模型和原始测试CSV后，在项目根目录运行（语义Python路径按本机替换）：

```powershell
python -m task2.embedding_predict --model task2/latest/model --test comp-5434-20261-project-task-2/test.csv --semantic-python .venv-semantic/Scripts/python.exe --work-dir task2_inference_round6 --output task2_prediction_round6.csv
python -m unittest discover -s task2/tests -v
```

下面保留历轮开发说明；其中outputs路径是当时本地训练产物，未随Git上传。

---

# COMP5434 Task 2：评论评分预测

使用 Dask 实际完成多进程特征任务与训练矩阵运算，评分模型和优化器由本项目实现。
所有命令均在项目根目录执行。代码包含详细中文注释，每阶段工作见 docs/阶段记录.md。

## 环境

本机已验证 Python 3.9.13、NumPy 1.21.5、pandas 1.4.4、SciPy 1.9.1、
scikit-learn 1.0.2、Dask/distributed 2022.7.0，单机三个工作进程。
建议 Python 3.9 独立环境，不要直接使用 Task 1 的 Python 3.12 环境安装这些旧版本。

```powershell
python -m pip install -r task2/requirements.txt
python -m unittest discover -s task2/tests -v
```

本机命令中的 `python` 对应 `D:/ProgramData/Anaconda3/python.exe`。
本机旧环境缺 six，项目内的 .local_deps 提供现有纯 Python 依赖副本及许可证；
正常安装 requirements 后不会使用该兼容路径，也不需要该目录。
不要求 Java、Spark、GPU 或外部云服务。

## 数据

默认输入 `comp-5434-20261-project-task-2/`，包含 train.csv、test.csv、sampleSubmission.csv。
CSV 约 2 GB、训练记录 7,684,645 条，不能简单整表读入内存。
标题/正文缺失按空串处理。显式固定空值词表，保留普通文本 "None"，避免 pandas 版本差异。

## 完整复现步骤

输出目录默认要求不存在，避免覆盖实验；本机已有输出，重跑请指定新的输出目录。

```powershell
python -m task2.audit --output task2/audit_compatible
python -m task2.prepare --workers 3 --chunk-size 50000
python -m task2.target_stats
python -m task2.optimize_cache
python -m task2.train --epochs 12
```

prepare 和 target_stats 支持分别用 --output / --cache 指定新的缓存路径，
后续训练通过 --cache 指向该缓存。缓存目录需要约 6–7 GiB 额外空间，另加原始数据和审计结果。
CSR 磁盘映射优化增加约 3.6 GiB，保留压缩缓存作为数值核对来源。
原 CSV 评论可能含换行，主进程流式解析合法记录后再向 Dask 提交有界的特征任务。
缓存是可重建的中间产物，不属于源码；最终打包代码时无需包含数 GiB 的缓存。

开发比较完成后，根据 outputs/development/*/summary.json 中的验证分数与最佳迭代次数，
选择方案，再训练全部数据。推荐由 finalize 自动读取结果并保存选择依据：

```powershell
python -m task2.finalize
python -m task2.predict --model task2/outputs/final/model --test comp-5434-20261-project-task-2/test.csv --output task2/outputs/reproduced_submission.csv
python -m task2.verify
python -m task2.benchmark
python -m task2.report
```

模型包包含权重及三个评分统计文件，独立预测不依赖缓存或原始训练评分。
prediction 计时包含模型加载、读取 CSV、全部特征转换和预测，不包含 CSV 写盘。
输出为 id、rating 两列，连续评分截断到 [1,5]，不四舍五入。

## 验证协议

- 评论身份、文本和时间构成的稳定哈希 mod 100 < 2 作为外层验证。
- 相同评论组不跨训练/验证，id 只对齐记录，不参与特征。
- 训练集约 98%，验证集约 2%，不按时间切分，因此不宣称未来预测能力。
- 外层验证标签不参与用户/商品统计、全局先验或优化。
- 训练行的评分统计再按评论组五折交叉拟合，排除自身整折；先验也排除该折。
- 测试行使用全量训练统计。未知标识符回退为训练先验并设置冷启动标记。
- 报告整体 RMSE，并另外报告用户和商品均有训练历史的 warm 子组与 cold 子组。
- 使用验证分数选择方案和轮数，因此这些是开发分数，不是独立测试或 Kaggle 分数。

## 算法与并行职责

HashingVectorizer 只做标题、正文一元/二元词组的通用哈希变换，不调用现成预测模型。
标题 2^15 维、正文 2^18 维，分别 L2 归一化，标题权重 1.5。
另外加入购买、票数、文本长度、时间等固定尺度字段。
元数据模型使用用户/商品/父商品平滑均值、支持数和冷启动标记。

自定义 Ridge 模型最小化平方误差加 L2 正则，通过预条件共轭梯度求解正规方程。
每次迭代完整遍历训练数据一次，工作进程计算 X_k.T(X_k p)，主进程按固定顺序归约。
不构造稠密 X.T X，工作进程只缓存有限数量的稀疏分块。
训练历史中的 iteration 是一次完整算子遍历，不是 mini-batch SGD 的 epoch；
报告逐轮遍历时间、统计初始化时间和总训练时间。

## 效率实验

```powershell
python -m task2.benchmark
```

相同前 12 分块、相同方向向量，比较 1 与 3 个工作进程，每配置三次。
这是训练核心算子的效率比较，不等于全量端到端训练加速比；保留实测结果，允许并行更慢。
清理 Python LRU，但操作系统文件缓存和背景负载无法完全控制。
最终总离线时间定义为文本缓存准备 + 额外映射转换 + 全量目标统计 + 全量模型训练，
不包含模型开发比较或可选数据审计；各项必须分别列出，避免混淆缓存热启动与完整重建。

## 关键文件

- audit.py：固定解析规则、数据审计、分组哈希。
- features.py / prepare.py：Dask 文本缓存和受限 LRU。
- target_stats.py：交叉拟合平滑统计及冷启动回退。
- train.py：分布式矩阵算子、自定义 PCG、验证和硬件记录。
- optimize_cache.py：CSR 压缩数组与映射数组逐元素一致性检查。
- export.py / predict.py：独立模型包与预测。
- finalize.py：按已完成的开发结果自动选择并训练全量模型。
- verify.py / report.py：逐条数据和提交核对、中文实验总结。
- benchmark.py：相同算子与不同并行配置的对比。
- tests/test_core.py：标签泄漏、平滑统计、PCG 与直接解、特征确定性检查。

## 本次已完成结果（2026-10-19）

| 方案 | 开发验证 RMSE |
|---|---:|
| 全局均值 | 1.516552 |
| metadata | 1.470608 |
| text + 基础字段 | 1.189914 |
| hybrid | 1.188911 |

最终 hybrid 已用全部 7,684,645 条记录训练 12 次完整遍历迭代。
独立模型包位于 outputs/final/model，提交预测位于 outputs/final/submission.csv。
六项核心测试、全量缓存/审计核对、独立预测与 CSV 一致性检查通过，尚未上传 Kaggle。
精确效果、硬件、每轮时间、总离线口径、推理和并行实验见 docs/实验总结.md。

## 2026-11-05 优化

新增两条相互区分的路线：

1. 从旧权重 warm-start，重启 PCG 搜索方向，新增 24 次完整遍历。开发 RMSE 从
   1.188911 降至 1.187962。不能把 warm-start 12+24 说成等同冷启动 36 轮。
2. 精确星级标题显示约 80% 一致、另四档各约 5% 的评分分布。基于“约 25% 均匀
   随机替换”假设校正评分收缩，参数只从训练统计估计，不使用测试标签或 Kaggle 分数拟合。

校正版的原评分验证 RMSE 为 1.228438，反而高于未校正的 1.187962。
独立标题代理 RMSE 为 0.261856，且假设下的潜在 RMSE 风险估计为 0.684771。
**标题代理和假设风险估计都不等于干净测试实测分数，不能将 0.684771 宣称为 Kaggle 成绩。**
保留常规版和校正版做线上对照，已提交的旧版本不变。

```powershell
python -m task2.noise_audit
python -m task2.train --variants hybrid --epochs 24 --initial-weights task2/outputs/development/hybrid/best_weights.npz --evaluate-every 6 --output task2/outputs/optimization_20261105/converged
python -m task2.calibration --include-converged
python -m task2.finish_optimization
```

以上为首次执行命令。本机已完成的输出应保留；不要重复覆盖同名目录。
独立校正模型包位于 outputs/optimization_20261105/full/model_calibrated/，使用原 predict 命令
指定该目录即可重生成校正预测；旧 model 包不受影响。新校正参数显式存于 model.json。
统一上传候选见项目根目录 submissions/20261105/，时间和限制见 docs/优化结果-20261105.md。

用户后续报告：task2_optimized_calibrated.csv 的 Kaggle RMSE 为 **0.64452**，
旧方案为 0.78159，相对下降约 17.54%。该校正版是当前已知最佳候选；
此数值为用户提供的线上反馈，不是前述标题代理或噪声模型风险估计。

## 第二轮优化（2026-11-05）

基于已获 0.64452 的第一轮校正模型，新增规范化标题、正文、文本对及用户-商品交互
查找统计。Dask 三进程实际处理全部训练记录并生成不含标签的键，随后用原训练部分
标签归约，所有原外层验证行被排除。零历史键的该项残差为零。
收缩残差组合器在组号 0/1 两半对称交叉拟合，再用全验证元数据拟合最终低维组合器；
最终查找表才使用全部训练数据。原模型的验证选择历史不变，不声称这是独立测试。

```powershell
python -m task2.lookup_features
python -m task2.lookup_train
python -m task2.lookup_context
python -m task2.lookup_predict --model task2/outputs/round2_20261105/model_context --test comp-5434-20261-project-task-2/test.csv --output task2/outputs/round2_20261105/reproduced_context.csv
```

这些为首次运行命令，本机已完成输出应保留。新模型包无需训练缓存或训练标签即可预测。
上下文候选的原评分验证 RMSE=1.237777、标题代理 RMSE=0.135438、
噪声模型风险 RMSE 估计=0.662360，基线三项分别为 1.228438、0.261856、0.684771。
**原评分 RMSE 仍变差，代理和模型估计不等于新的 Kaggle 成绩。**
独立预测文件逐字节一致，详见根目录 docs/第二轮优化记录-20261105.md。

用户随后反馈该第二轮上下文版线上 RMSE=0.62238，成为当前已知最佳，0.64452版本仍保留。

## 第三轮：非线性组合（2026-11-05）

新增35维结构化输入与自主平方损失直方图提升树，从线性组合开始拟合非线性残差。
元目标为未裁剪的逆均匀替换通道评分，输出才限制1–5；最终不再重复噪声校正。
开发组内部80/20选择结构、轮数和幅度，冻结后检查另一组；基础模型的原验证历史如实保留。
显式星级已作为输入，不使用标题代理分数挑选候选。确认组全体与无星级标题子组的风险均改善，
但风险仍依赖噪声假设，原评分RMSE变差，不是新Kaggle成绩。

```powershell
python -m task2.round3_features
python -m task2.round3_train
python -m task2.round3_predict --model task2/outputs/round3_20261105/model --test comp-5434-20261-project-task-2/test.csv --output task2/outputs/round3_20261105/reproduced.csv
python -m tools.finish_task2_round3
```

以上是首次执行命令，本机已存在的实验目录不得重复覆盖。
独立模型包包含继承依赖，无需训练评分和旧缓存即可预测。
新候选：submissions/20261105_task2_round3/task2_nonlinear_residual.csv。
完整报告：docs/Task2第三轮优化结果-20261105.md；当前线上最佳仍是用户报告的0.62238。

用户后续报告第三轮非线性版线上RMSE=0.60134，更新为当前已知最佳，旧0.62238保留。

## 第四轮：稳健词辅助与追加残差（2026-11-05）

参考原词模型计算一次固定Huber权重，delta=1，再由Dask三进程自主PCG拟合加权Ridge。
辅助评分不直接视为干净标签，不盲目套用旧校正作最终输出；原尺度/逆通道尺度/相对差
作为辅助特征，由追加的浅层树选择是否信任。保留“不加辅助、仅继续残差”的内部对照。
配置在组号0内部选择，冻结后在组号1确认；历史验证复用与噪声假设限制仍成立。

```powershell
python -m task2.robust_text
python -m task2.round4_features
python -m task2.round4_train --stage develop
python -m task2.round4_uncertainty
python -m task2.robust_text --mode full
python -m task2.round4_features --mode full
python -m task2.round4_train --stage final
python -m task2.round4_predict --model task2/outputs/round4_20261105/model --test comp-5434-20261-project-task-2/test.csv --output task2/outputs/round4_20261105/reproduced.csv
python -m tools.finish_task2_round4
```

以上为首次执行顺序，本机已有目录不重复覆盖。每条命令完成后再执行下一条。
确认组原评分RMSE从1.242751降至1.239266，噪声假设风险估计从0.636006降至0.631268，
无星级标题子组风险从0.671779降至0.666796；都不是新的Kaggle分数。
18项回归测试、独立CSV一致性及旧文件哈希检查通过。Task1本轮不变。
新候选：submissions/20261105_task2_round4/task2_robust_refinement.csv，当前已知线上最佳0.60134不覆盖。
阶段和结果见根目录 docs/Task2第四轮优化记录-20261105.md、docs/Task2第四轮优化结果-20261105.md。

用户后续报告第四轮线上RMSE=**0.59347**，更新为当前已知最佳，旧0.60134保留。

## 第五轮：上下文语义与显式噪声通道（2026-11-05）

本轮新增固定的NLP Town五星BERT/Xenova ONNX语义输入，保留原Dask词回归与历史统计。
它提供五星logits，融合器自主实现噪声通道似然、解析梯度和非线性残差树。
没有在本机微调BERT，融合训练使用独立于标签/误差选出的57,797条记录。
原153,456条验证组的历史复用仍披露，不能把假设风险当成新Kaggle成绩。

**本候选使用外部预训练模型，比赛是否允许尚未核对。**
请在上传前核对规则；模型来源、固定版本、原模型卡、哈希与外部训练重叠未知情况都已保留。
课程自主实现与Dask部分不被替代成一次外部分类器调用，详见根目录第五轮阶段/结果文档。

本机实际环境：原训练Python为 `D:/ProgramData/Anaconda3/python.exe`；
新增推理Python为 `C:/Users/Leo/.cache/comp5434-round5/venv/Scripts/python.exe`。
可在其他机器用Python3.12新建隔离环境，安装 `task2/requirements-semantic.txt`；
原训练环境仍依照原requirements安装，不覆盖既有NumPy/SciPy/Dask版本。

以下为首次执行顺序，已有输出不要重跑覆盖。执行目录为项目根目录。

```powershell
$task2Py = 'D:/ProgramData/Anaconda3/python.exe'
$semanticPy = 'C:/Users/Leo/.cache/comp5434-round5/venv/Scripts/python.exe'
& $semanticPy -X utf8 -m task2.semantic_prepare --stage download
& $task2Py -X utf8 -m task2.semantic_prepare --stage extract
& $task2Py -X utf8 -m task2.semantic_prepare --stage subset
& $semanticPy -X utf8 -m task2.semantic_infer --model task2/outputs/round5_20261105/external_model --input task2/outputs/round5_20261105/semantic_development_text.csv --output task2/outputs/round5_20261105/semantic_development.npz --provider cpu --batch-size 16 --max-length 128 --dynamic-length
& $semanticPy -X utf8 -m task2.semantic_infer --model task2/outputs/round5_20261105/external_model --input task2/outputs/round5_20261105/test_text.csv --output task2/outputs/round5_20261105/semantic_test.npz --provider cpu --batch-size 16 --max-length 128 --dynamic-length
& $task2Py -X utf8 -m task2.semantic_train --stage develop
& $task2Py -X utf8 -m task2.semantic_uncertainty
& $task2Py -X utf8 -m task2.semantic_train --stage final
& $task2Py -X utf8 -m task2.semantic_predict --model task2/outputs/round5_20261105/model --test comp-5434-20261-project-task-2/test.csv --semantic-python $semanticPy --work-dir task2/outputs/round5_20261105/independent_inference --output task2/outputs/round5_20261105/reproduced.csv
# 如需复现本轮扩展实验，先执行下方扩展步骤，再运行finish_task2_round5。
```

从零训练需先完成前四轮，以获得固定词模型、统计表与结构化验证特征。
但独立推理只需第五轮完整 `model/` 包、原始test.csv、两个Python环境和项目源码，
不需要训练评分、词特征缓存或旧提交文件。

新机器可仅执行上面的 `semantic_predict`，改为新的工作目录和输出文件名。
工作目录保留语义推理断点；输入/模型/批量参数改变时断点校验会拒绝混用。
最终固定批量16、最长128、CPU量化与长度排序；量化输出随批组成存在差异，不能随意换成GPU路径。
本机相同环境验证逐字节重现；其他硬件/运行库不承诺浮点末位逐字节一致。

候选目录：`submissions/20261105_task2_round5/`；
文件名 `task2_semantic_noise_fusion.csv`，包含1万条连续评分，上传仍为id,rating。
实际结果与检查状态以根目录 `docs/Task2第五轮优化结果-20261105.md` 和候选manifest为准。

第五轮另增加有历史专家：测试用户/商品/父商品均已有训练历史，普通验证约44%用户无历史。
在查看语义效果之前预设以三种身份均有历史的内部子集风险选型，保留全体结果与同条件去语义消融。
缺历史记录由全体通道回退；门控只读取统计支持数，不读取测试标签。

本轮在首次确认的只读消融后另登记扩展审计，原预选tree400保留。
额外区间9,447条不参与本轮融合训练，只比较已保存的confirm通道与tree400，各幅度1；
额外审计风险更低且组级差异95%区间上界小于0，才额外发布通道候选。
以下需接在上述独立推理之后执行；先完成额外审计，再整理manifest，使模型清单完整。

```powershell
& $task2Py -X utf8 -m task2.semantic_extra_audit --stage prepare
& $semanticPy -X utf8 -m task2.semantic_infer --model task2/outputs/round5_20261105/external_model --input task2/outputs/round5_20261105/extra_text.csv --output task2/outputs/round5_20261105/semantic_extra.npz --provider cpu --batch-size 16 --max-length 128 --dynamic-length
& $task2Py -X utf8 -m task2.semantic_extra_audit --stage evaluate
$extraAudit = Get-Content -LiteralPath task2/outputs/round5_20261105/extra_audit.json -Raw | ConvertFrom-Json
if ($extraAudit.release_channel_as_additional_candidate) {
    & $task2Py -X utf8 -m task2.semantic_channel_predict --model task2/outputs/round5_20261105/model --test comp-5434-20261-project-task-2/test.csv --semantic-predictions task2/outputs/round5_20261105/semantic_test.npz --output task2/outputs/round5_20261105/channel_submission.csv
    & $task2Py -X utf8 -m task2.semantic_channel_predict --model task2/outputs/round5_20261105/model --test comp-5434-20261-project-task-2/test.csv --semantic-predictions task2/outputs/round5_20261105/independent_inference/semantic.npz --output task2/outputs/round5_20261105/channel_reproduced.csv
}
& $task2Py -X utf8 -m tools.finish_task2_round5
& $task2Py -X utf8 -m tools.finish_task2_round5_extra
```

通道候选的独立复现复用刚刚独立生成的语义输出，因此本阶段时间不含再次运行BERT。
它从原始test.csv重建全部旧词/历史特征，校验语义文件文本指纹和包内网络指纹，
不用训练缓存即可预测。两条路线共用同一个完整模型包；`model.json`仍选择预选树，
`channel_candidate.json`说明扩展通道路线，使用`semantic_channel_predict`调用。
如果扩展审计未达条件，就不生成通道提交文件；审计报告仍保留结果，不继续扫描区间找优势。

本机最终结果：扩展审计没有达到额外发布通道条件，保留原预选语义残差树，只有一个新上传候选。
冻结确认风险估计0.625333→0.584974，额外复核0.590199→0.556741；均不是Kaggle实测。
24项回归测试通过，完整原始CSV独立复现逐字节一致，旧0.59347提交保留。

用户2026-11-08反馈第五轮线上RMSE=**0.54604**，成为当前已知最佳。
同日用户转述老师允许外部预训练模型，条件是自己理解模型。旧包的规则状态是历史记录，
不修改已校验模型哈希；新实验使用本次确认，理解材料见 `docs/BERT原理与本项目答辩指南.md`。

## 第六轮：768维CLS与自主任务预测头（2026-11-08）

第五轮使用冻结网络的五维logits。本轮额外取最后一层、pooler之前的CLS768维，
结合已有59维、第五轮评分1维，共828维；只训练头，不修改BERT内部权重。
测试身份都有历史，因此83,047条匹配支持条件的原验证记录作为本轮样本；未来冷启动输入回退第五轮。
先登记岭回归/噪声线性头/自主MLP对照，内部选择后冻结确认，报告同预测头移除CLS的消融。
历史验证复用、量化批组成和噪声假设局限均保留说明。

下面是首次执行顺序，已有同名实验目录不能覆盖。原数值Python按requirements.txt安装；
Python3.12语义环境按requirements-embedding.txt安装，后者新增onnx仅用于图准备和验证。

```powershell
$task2Py = 'D:/ProgramData/Anaconda3/python.exe'
$semanticPy = 'C:/Users/Leo/.cache/comp5434-round5/venv/Scripts/python.exe'
& $semanticPy -m pip install -r task2/requirements-embedding.txt
& $semanticPy -X utf8 -m task2.embedding_prepare
& $semanticPy -X utf8 -m task2.embedding_infer --encoder task2/outputs/round6_20261108/encoder.onnx --tokenizer task2/outputs/round5_20261105/external_model/tokenizer.json --input task2/outputs/round6_20261108/development_text.csv --output task2/outputs/round6_20261108/development.npz
& $semanticPy -X utf8 -m task2.embedding_infer --encoder task2/outputs/round6_20261108/encoder.onnx --tokenizer task2/outputs/round5_20261105/external_model/tokenizer.json --input task2/outputs/round6_20261108/test_text.csv --output task2/outputs/round6_20261108/test.npz
& $task2Py -X utf8 -m task2.embedding_train --stage develop
& $task2Py -X utf8 -m task2.embedding_uncertainty
& $task2Py -X utf8 -m task2.embedding_structure_ablation
& $task2Py -X utf8 -m task2.embedding_train --stage final
& $task2Py -X utf8 -m task2.embedding_predict --model task2/outputs/round6_20261108/model --test comp-5434-20261-project-task-2/test.csv --semantic-python $semanticPy --work-dir task2/outputs/round6_20261108/independent_inference --output task2/outputs/round6_20261108/reproduced.csv
& $task2Py -X utf8 -m tools.finish_task2_round6
```

完整从零训练仍需按前几轮生成原词模型、统计和第五轮参数；本轮相应源码使用固定日期目录。
独立推理只需第六轮model包、源码、原始测试CSV和两个环境，不需要训练评分或缓存。
新机器可单独运行embedding_predict并指定新的输出/工作目录。首次输出不复用语义断点，
本机同环境核对逐字节一致；其他硬件浮点末位不保证一致。
如冻结确认未改善，final会拒绝创建推荐模型，应保留失败记录而不是修改确认参数。
真实最终状态以根目录第六轮结果报告和候选清单为准。

第六轮已完成：选中MLP十二轮，线性锚点正则1000，融合幅度1。
本地风险估计0.543579→0.529699，29项测试通过，原始CSV独立复现逐字节相同。
提交文件 `submissions/20261108_task2_round6/task2_cls_adapted.csv`，完整参数包在对应outputs/model。
这不是新的Kaggle成绩，当前已知线上最佳仍0.54604。
