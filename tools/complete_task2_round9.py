"""当前轮确认后的顺序流水线；门槛失败立即停止，不发布较差候选。

仅运行已经冻结的配置，不重新选型、不操作Kaggle或GitHub。
长任务完成后自动衔接下一阶段，避免等待人工逐条启动导致GPU空闲。
每步实际起止、返回码写入JSON；失败时保留现场和所有历史文件。
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import task2
from task2.audit import ROOT,sha256,write_json
from task2.partial9_protocol import OUT,FOLDER


def main():
    ledger=OUT/"completion_pipeline.json"
    if ledger.exists():raise FileExistsError("流水线记录已存在，请检查现场，不自动覆盖或重训。")
    for name in ["neural_final.npz","submission.csv","reproduced.csv"]:
        if (OUT/name).exists():raise FileExistsError(OUT/name)
    home=Path.home()/".cache"
    cuda=FOLDER/"venv/Scripts/python.exe"
    cpu=home/"comp5434-english/venv/Scripts/python.exe"
    bert=home/"comp5434-round5/venv/Scripts/python.exe"
    numeric=Path(sys.executable)
    state={"status":"running","steps":[],"automatic_kaggle_submission":False,"automatic_git_push":False}
    journal=ROOT/"docs/Task2第九轮优化记录-20261114.md"
    def note(message):
        with journal.open("a",encoding="utf-8") as stream:stream.write("\n"+message+"\n")
    def run(label,command):
        """以参数列表启动子进程，不把路径/配置拼成可执行shell表达式。"""
        row={"name":label,"started_utc":datetime.now(timezone.utc).isoformat()}
        state["steps"].append(row);write_json(ledger,state)
        note("### 流水线："+label+"\n\n开始UTC："+row["started_utc"]+"。")
        print("第九轮流水线开始",label,flush=True);began=time.perf_counter()
        result=subprocess.run([str(p) for p in command],cwd=ROOT)
        row.update(seconds=time.perf_counter()-began,returncode=result.returncode,
            finished_utc=datetime.now(timezone.utc).isoformat());write_json(ledger,state)
        note("结束UTC："+row["finished_utc"]+f"；实际{row['seconds']:.2f}秒，返回码{row['returncode']}。")
        if result.returncode:raise RuntimeError("流水线阶段失败："+label)
        print("第九轮流水线完成",label,flush=True)
    def module(python,name,*args):return [python,"-X","utf8","-m",name,*args]
    try:
        choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
        run("冻结配置确认",module(numeric,"task2.partial9_fusion","--stage","confirm"))
        proof=json.loads((OUT/"confirmation.json").read_text(encoding="utf-8"))
        if not choice["internal_improved"] or proof["candidate"]["noise_model_risk"]["latent_mse_estimate"]>=proof["baseline"]["noise_model_risk"]["latent_mse_estimate"]:
            raise RuntimeError("内部/确认未共同胜出，保持第八轮推荐文件。")
        note("固定确认风险RMSE："+str(proof["baseline"]["noise_model_risk"]["latent_rmse_estimate"])+" → "+
            str(proof["candidate"]["noise_model_risk"]["latent_rmse_estimate"])+"。通过内部/确认双重门槛，保持原冻结配置，进入最终训练；线上分数未知。")
        # 配置来自已经冻结的JSON，不根据确认消融结果替换目标、轮次或融合。
        run("最终83047行适配",module(cuda,"task2.partial9_train","--stage","final",
            "--learning-rate",str(choice["learning_rate"]),"--epochs",str(choice["epochs"]),
            "--objective",choice["objective"],"--eval-max-length",str(choice["max_length"])))
        run("最终融合参数",module(numeric,"task2.partial9_fusion","--stage","fit-final"))
        run("实际适配权重部署核验",module(cuda,"tools.check_partial9_deployment"))
        run("自包含模型包",module(numeric,"tools.finish_task2_round9","--stage","pack"))
        model=FOLDER/"model";cache=FOLDER/"test_cpu"
        text=ROOT/"task2/outputs/round6_20261108/test_text.csv"
        run("测试集全新CPU前缀与适配输出",module(cpu,"task2.partial9_infer","--model",model/"adapter",
            "--input",text,"--cache-folder",cache,"--output",OUT/"neural_test.npz"))
        run("候选CSV生成",module(numeric,"task2.partial9_fusion","--stage","final"))
        work=FOLDER/"independent";work.mkdir(exist_ok=True)
        old=ROOT/"task2/outputs/round8_20261114/independent"
        names=["input.csv","embedding.npz","embedding.embedding.npy","embedding.logits.npy","embedding.state.json",
            "english.npz","english.embedding.npy","english.logits.npy","english.state.json"]
        # 只复用无标签冻结编码。旧评分CSV和旧计时文件不属于该清单。
        for name in names:
            target=work/name
            if target.exists():
                if sha256(target)!=sha256(old/name):raise ValueError("独立冻结缓存来源不一致："+name)
            else:shutil.copy2(old/name,target)
        if sha256(work/"input.csv")!=sha256(text):raise ValueError("原始独立文本与首次测试编码输入不一致。")
        if json.loads((cache/"state.json").read_text(encoding="utf-8"))["completed"]!=10000:
            raise ValueError("测试前缀尚未完整完成，禁止复用。")
        dest=work/"prefix";dest.mkdir(exist_ok=False)
        for name in ["tokens.json","tokens.npy","lengths.npy","offsets.npy","id.npy","hidden.npy","state.json"]:
            # 只有完整的只读表示采用同卷硬链接，其余小元数据独立复制。
            if name=="hidden.npy":
                try:os.link(cache/name,dest/name)
                except OSError:shutil.copy2(cache/name,dest/name)
            else:shutil.copy2(cache/name,dest/name)
        run("原始CSV独立特征与评分重建",module(numeric,"task2.round9_predict","--model",model,
            "--test",ROOT/"comp-5434-20261-project-task-2/test.csv","--semantic-python",bert,
            "--english-python",cpu,"--work-dir",work,"--output",OUT/"reproduced.csv"))
        run("CSV格式与独立复现核验",module(numeric,"tools.finish_task2_round9","--stage","verify"))
        run("实际结果报告",module(numeric,"tools.report_task2_round9"))
        state["status"]="complete";write_json(ledger,state)
        print("第九轮本地交付流水线全部完成，Git副本仍需单独验证。",flush=True)
    except Exception as error:
        note("流水线停止："+str(error)+"。所有已完成产物与旧文件保留。")
        state.update(status="failed",error=str(error));write_json(ledger,state);raise


if __name__=="__main__":main()
