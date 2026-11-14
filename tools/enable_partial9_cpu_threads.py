"""为尚未发布的本轮包启用已通过逐位一致性核验的CPU线程配置。

权重和所有其他参数逐文件核对不变，只更新执行配置与完整性清单。
首次预测进程已读取旧配置，继续使用单线程；后续全量独立复现检查两者评分。
"""
import json
import task2
from task2.audit import ROOT,sha256,write_json
from task2.partial9_protocol import OUT,FOLDER


def main():
    audit=json.loads((OUT/"cpu_threads_audit.json").read_text(encoding="utf-8"))
    if not audit["bit_identical"] or audit["top_threads"]!=4:print("保持单线程。",flush=True);return
    model=FOLDER/"model";path=model/"file_integrity.json"
    before=json.loads(path.read_text(encoding="utf-8"))
    for name,value in before.items():
        if sha256(model/name)!=value["sha256"]:raise ValueError("线程调整前模型完整性不匹配："+name)
    runtime=model/"adapter/runtime.json";previous=json.loads(runtime.read_text(encoding="utf-8"))
    if previous.get("top_threads",1)!=1:raise RuntimeError("线程配置已改变，不重复更新。")
    write_json(OUT/"runtime_before_cpu_threads.json",previous);write_json(OUT/"integrity_before_cpu_threads.json",before)
    write_json(runtime,{**previous,"top_threads":4})
    after={str(p.relative_to(model)).replace("\\","/"):{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(model.rglob("*")) if p.is_file() and p!=path}
    if any(after[name]!=value for name,value in before.items() if name!="adapter/runtime.json"):raise ValueError("线程调整意外改变了其他模型文件。")
    write_json(path,after)
    ref=OUT/"model_reference.json";metadata=json.loads(ref.read_text(encoding="utf-8"))
    metadata.update(logical_bytes=sum(p.stat().st_size for p in model.rglob("*") if p.is_file()),cpu_top_threads=4,cpu_thread_audit_sha256=sha256(OUT/"cpu_threads_audit.json"))
    write_json(ref,metadata)
    with (ROOT/"docs/Task2第九轮优化记录-20261114.md").open("a",encoding="utf-8") as stream:
        stream.write("\n## CPU最后层线程执行核验\n\n32条256token真实适配输入的单/四线程logits和CLS逐位相同。该次实测26.337/12.521秒；为后续预测启用四线程，首次已启动预测保留单线程。仅更新runtime和完整性清单，所有权重/其他参数SHA不变；全量独立复现还会验证评分一致。前缀仍OpenVINO四线程，前缀/最后层顺序执行。\n")
    print("CPU执行配置已更新，权重/其他参数全部不变。",flush=True)


if __name__=="__main__":main()
