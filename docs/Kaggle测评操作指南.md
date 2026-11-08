# Kaggle 测评操作指南

准备日期：2026-10-27（Asia/Shanghai）。本指南说明人工上传步骤；已根据用户反馈的 Task 1 名称版格式错误更正。

## 1. 比赛入口与队伍

- Task 1：https://www.kaggle.com/t/d6784c43253c4aaf8659a4a6eb17c697
- Task 2：https://www.kaggle.com/t/b818816f6d244be7afc696c87bf48fbf

登录 Kaggle 后分别打开两个课程邀请链接，确认比赛名称与数据相符。
如果尚未加入，点击 Join Competition 并阅读、接受规则。已经加入则跳过。
两个比赛分别核对 Team / My Team 页，队名应与 Canvas 的课程团队名单一致。
已有团队的成员加入原团队，不创建另一个同名队伍；团队组织按各比赛页面规则执行。

私有比赛页面无法由本次公开网页工具读取，具体按钮、Evaluation、限额与截止时间以登录后页面为准。
Kaggle 官方一般提交说明：https://www.kaggle.com/docs/competitions 。

## 2. 上传文件

所有路径相对项目根目录 F:/POLYU COURSES/Semester1/COMP5434/project。

| 任务 | 首次候选文件 | 格式 | 记录数 |
|---|---|---|---:|
| Task 1（应上传） | task1/outputs/baseline/forest/submission_integer.csv | id,label；整数 1–4 | 3,300 |
| Task 1 名称解释版（不要上传） | task1/outputs/baseline/forest/submission_weather.csv | id,label；天气名称 | 3,300 |
| Task 2 | task2/outputs/final/submission.csv | id,rating；连续评分 | 10,000 |

这些都是预测文件，不需要为本次上传重新训练。不要上传原始 train/test、原始提交样例、
OOF 预测、model.json 或整个代码压缩包替代预测 CSV。代码、报告、slides 按课程要求另行提交。

## 3. Task 1 编码确认

课程 PDF 描述整数标签，但已下载的 sample_submission.csv 使用 Rainy 等天气名称。
最初据此推荐名称版的判断已被实际线上错误否定：用户反馈 Kaggle 返回
`Evaluation Exception: The input string 'Rainy' was not in a correct format.`。
评测器需要数值标签。当前应上传 submission_integer.csv，不再试传名称版。

固定映射：1 Sunny、2 Cloudy、3 Rainy、4 Snowy。
两份文件代表完全相同的预测，不能把它们当作两个不同模型。
本次名称版在格式解析阶段失败，尚未得到模型分数；换用已有整数版即可，无需重训。

应上传整数版的前三条：

```csv
id,label
100000,3
100001,3
100002,2
```

## 4. Task 1 上传

1. 进入 Task 1 比赛，点击 Submit Predictions / Make a Submission（以页面为准）。
2. 在文件选择框中选择 submission_integer.csv；Windows 可将完整路径粘贴到“文件名”栏。
3. 若有提交说明栏，填写 `Task1 custom RF | 80 trees | depth16 leaf3 | local CV Macro-F1=0.915223 | integer labels`。
4. 核对文件名和比赛，再点击 Submit。
5. 在 Submissions 查看处理状态，处理完成后记录 Score、提交说明、时间和提交编号（若可见）。
6. Task 1 指标 Macro-F1 越高越好；本地 CV 0.915223 仅供参照，不是 Kaggle 承诺分数。

## 5. Task 2 上传

1. 打开 Task 2 比赛，确认其指标为 RMSE。
2. 点击 Submit Predictions / Make a Submission，选择 task2/outputs/final/submission.csv。
3. 提交说明可填写 `Task2 custom ridge-PCG hybrid | Dask 3 workers | 12 iterations | local validation RMSE=1.188911`。
4. 点击 Submit，等待评分，记录提交时间、说明、编号与 Score。
5. Task 2 指标 RMSE 越低越好。本地验证 1.188911 不等于 Kaggle 测试成绩。

输出是连续评分，不要使用 Excel 将其改成整数或另存为 XLSX。

```csv
id,rating
0,4.1332984837
1,4.1796060726
2,3.7564947410
```

## 6. 如何读结果

Public Score 来自公开测试子集，Private Score/最终排名在比赛结束后按剩余子集计算。
本地验证、公开榜、私有榜的数据不同，分数无需相同。只改变 CSV 编码不属于算法改进。
先完成两个任务各一次有效提交，再根据结果判断下一步，不把每日次数全用在盲试上。

课程文档写明每天最多 10 次提交，通常额度由同一比赛的整个团队共享；实时剩余额度以页面为准。
课程要求每个比赛在结束前选定两个候选提交，最终以所选候选的私有榜表现确定名次。
在 Submissions 查找 Select / Use for final scoring 或相应选择控件，并核对比赛当前要求。
如果目前只有一个模型，先保留该有效提交；最终第二个候选应来自后续有效实验，不用仅改文件名的重复提交凑数。

课程文档时间：预测截止 2026-11-14 23:55；最终排名 2026-11-15 08:00；
代码/报告/slides 截止 2026-11-16 08:00。PDF 未注明时区，必须与 Kaggle Timeline 和 Canvas 核对。

## 7. 常见问题

| 情况 | 检查方式 |
|---|---|
| 看不到提交按钮 | 确认已登录、加入该比赛、接受规则，且尚在允许加入/提交的时间内 |
| 被要求验证账号 | 按 Kaggle 页面完成；不是所有账号都需要同样的验证 |
| 缺少列或列名错误 | Task 1 是 id,label；Task 2 是 id,rating，区分大小写 |
| 行数/ID 不符 | 不要传错任务；Task 1 为 3,300 条，Task 2 为 10,000 条 |
| 无法解析标签 | Task 1 核对名称/整数编码；Task 2 保持数值，不含空值 |
| 处理失败 | 展开错误详情，保留原文后排查，不反复提交同一文件 |
| 零分或异常低分 | 优先核对比赛、文件、标签编码和任务指标，再分析模型 |
| 没有每日额度 | 等页面提示的重置时间，团队统一安排提交 |

## 8. 本次本地文件检查证据

2026-10-27 已重新检查三份预测的列名、记录数、ID 顺序及标签范围，均符合本地测试集。

| 文件 | SHA-256 |
|---|---|
| Task 1 名称版 | f1148465472c5fc004d42cc3936c1f1d85adfd5bafb114453c74738164f80846 |
| Task 1 整数版 | 0506e4d295b42df48c09caf0d9a4feda0aa4e35dc312ca3f479b51e1181fa691 |
| Task 2 混合模型 | 044c22d6c533e05de661b396b1faf0fb9842c7feb781a806a14346bfd6a035b3 |

实际 Kaggle 成绩填写到同目录 Kaggle测评记录.md，当前不填写虚构分数。
