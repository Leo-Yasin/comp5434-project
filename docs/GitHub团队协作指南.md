# GitHub团队协作指南

整理日期：2026-11-08。适用于当前COMP5434项目。

## 1. 当前状态与共享范围

当前项目原先没有Git仓库，电脑已安装Git。现已准备根目录README和.gitignore，尚未创建远程仓库或上传文件。
推荐建立团队私有仓库，邀请本组队友，统一保存源码、中文方案、实验结论、提交CSV和指纹清单。

根目录.gitignore排除两个原始数据文件夹、task*/outputs、Task 2缓存、模型二进制、临时文件、
个人环境、凭据与根目录PDF。现有 `submissions/` 中的小型CSV和JSON清单保留。
GitHub普通Git仓库会阻止单文件超过100MiB；浏览器上传单文件上限25MiB。
原始训练CSV约2GB，模型包数百MB，应单独共享，不能直接把整个项目拖上网页。
来源：[GitHub大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 2. 创建远程私有空仓库

1. 登录GitHub，点击右上角“+”→“New repository”。
2. 仓库名建议 `comp5434-course-project`，描述填写两项任务及团队协作用途。
3. 可见性选Private。本指南按个人账号拥有仓库、邀请队友协作讲解。
4. 创建时不要勾选自动生成README、.gitignore或License，本地已有前两项，避免首次推送历史不一致。
5. 点击Create repository，复制HTTPS地址，例如 `https://github.com/YOUR_ACCOUNT/comp5434-course-project.git`。

## 3. 首次本地提交并推送

在PowerShell逐条执行；姓名、邮箱和仓库地址替换为自己的值。
本地user.name是提交显示名，不强制等于GitHub用户名。邮箱可以使用GitHub邮箱设置提供的noreply地址。

```powershell
Set-Location -LiteralPath 'F:\POLYU COURSES\Semester1\COMP5434\project'
git init -b main
git config user.name "你的提交显示名"
git config user.email "你的GitHub验证邮箱或noreply邮箱"
git status --short
git add --dry-run .
```

先检查预览：应有README、.gitignore、docs、task1/task2的源码/测试/说明、tools和submissions。
不应出现原始数据、outputs、缓存、.env或kaggle.json。确认后继续：

```powershell
git add .
git diff --cached --stat
git commit -m "docs: 整理Task1及Task2已有方案与五轮优化记录"
git remote add origin https://github.com/YOUR_ACCOUNT/comp5434-course-project.git
git push -u origin main
```

首次HTTPS推送可能触发Git Credential Manager浏览器登录，按页面完成授权。
若命令行要求认证，不要输入GitHub账号密码，也不要把令牌写进远程URL或仓库文件。
可按[GitHub认证说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github)选择凭据管理器、令牌或SSH。

推送后刷新GitHub页面，检查README各轮索引和提交CSV是否可打开。
来源：[GitHub导入本地代码指南](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 4. 邀请队友

仓库页面→Settings→Access区域的Collaborators→Add people→输入队友GitHub用户名→发送邀请。
队友接受后即可访问该私有仓库。这里是个人私有仓库协作者流程；组织仓库入口可能显示团队设置。
来源：[GitHub邀请协作者](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository)。

队友克隆：

```powershell
git clone https://github.com/YOUR_ACCOUNT/comp5434-course-project.git
Set-Location -LiteralPath comp5434-course-project
```

先看根目录README，再按任务说明准备数据和环境。仓库不含模型包，不能仅克隆后就直接运行最终预测。

## 5. 以后每一轮优化怎么记录

建议每轮独立分支、一次Pull Request。下面是未来第六轮的命名示例，不代表已经做过该轮。

```powershell
git switch main
git pull --ff-only
git switch -c task2/round6
```

这一轮至少一起保存：

| 内容 | 建议位置 |
|---|---|
| 起点、想验证的假设、模型改动及训练参数 | docs/Task2第六轮优化记录-实际日期.md |
| 训练/验证划分、风险假设、真实计时和消融结果 | 对应结果文档 |
| 中文注释源码、依赖与复现命令 | task2/、tools/和README |
| 小型最终预测文件和指纹清单 | submissions/实际日期_task2_round6/ |
| 实际线上分数、提交编号和时间 | docs/Kaggle测评记录.md |
| 模型下载位置、版本及SHA-256（模型另存） | 对应结果文档 |

用“待测评”表示没有线上成绩。本地噪声风险估计要单独标注，不能当成Kaggle成绩。
每阶段都可以小提交，轮次完成时再整合PR；不需要等到最后才补写过程。

```powershell
git status --short
git add docs task2 tools submissions README.md
git diff --cached --stat
git commit -m "feat(task2): 记录第六轮模型改动与验证结果"
git push -u origin task2/round6
```

打开GitHub仓库，创建该分支到main的Pull Request；说明问题、实际改变、对照指标、检查和限制。
请队友阅读，合并后其他成员回到main执行 `git pull --ff-only`。
若pull报告分叉或本地有未提交修改，先查看git status并处理，不通过force push覆盖队友提交。

团队最终选定一个版本后，可在包含该版本及结果的main提交上加标签，例如：

```powershell
git switch main
git pull --ff-only
git tag -a task2-round6 -m "Task2第六轮已合并版本"
git push origin task2-round6
```

以后队友可通过该标签查看当时的代码。已有前五轮没有原始逐轮Git提交，不能从当前目录
自动恢复每一轮当时的完整源码快照；现有脚本、文档、CSV和哈希能说明保留的历史证据。

## 6. 大型模型怎么给队友

把完整模型包放在团队私有网盘，或在私有仓库按版本建立Release并附模型压缩包。
模型包与任务README的版本对应，提供压缩包SHA-256及解压目标目录，队友下载后再独立预测。
不要只分发一个weights文件：第五轮模型依赖旧词模型、查找表、残差树、外部量化网络和分词器。
源数据仍由队友通过比赛下载；外部模型来源、规则待核对状态保留。
来源：[GitHub分发大文件建议](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 7. 本次准备记录

已新增根目录README、根目录.gitignore与本指南。没有修改原模型、提交CSV或线上成绩。
尚未在项目根目录初始化Git，未创建远程仓库、邀请队友或推送。
已用隔离的临时Git目录检查真实忽略规则：预计纳入130个文件、约1.65MiB，最大单文件约185KiB；
原始数据、outputs与缓存未出现在待上传列表。根目录README和本指南的本地链接全部可找到。
该预览没有创建项目根目录的.git，也没有修改原提交文件。详细本地预览在tmp/github_upload_preview.json。
