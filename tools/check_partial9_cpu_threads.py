"""比较实际最终最后层的CPU单/四线程；只有逐位相同且更快才启用四线程。"""
import json
import time
import numpy as np
import torch
from task2.partial9 import load_top,FOLDER,OUT
from task2.partial9_protocol import run_name
from task2.audit import write_json


def main():
    choice=json.loads((OUT/"frozen_selection.json").read_text(encoding="utf-8"))
    sample=np.load(FOLDER/("audit_batch_256.npz" if choice["max_length"]==256 else "audit_batch.npz"))
    model=load_top(FOLDER/run_name("final",choice)/f"epoch_{choice['epochs']}.pt",FOLDER/"config").eval()
    values={};timings={}
    for threads in [1,4]:
        torch.set_num_threads(threads);parts=[];embeddings=[];began=time.perf_counter()
        with torch.no_grad():
            for first in range(0,len(sample["hidden"]),16):
                logits,cls=model(torch.from_numpy(sample["hidden"][first:first+16]),torch.from_numpy(sample["mask"][first:first+16]).bool())
                parts.append(logits.numpy());embeddings.append(cls.numpy())
        values[threads]=(np.concatenate(parts),np.concatenate(embeddings));timings[str(threads)]=time.perf_counter()-began
    exact=all(np.array_equal(a,b) for a,b in zip(values[1],values[4]))
    enabled=4 if exact and timings["4"]<timings["1"] else 1
    proof={"rows":len(sample["hidden"]),"max_length":choice["max_length"],"labels_used":False,
        "logit_max_error":float(np.abs(values[1][0]-values[4][0]).max()),"cls_max_error":float(np.abs(values[1][1]-values[4][1]).max()),
        "bit_identical":exact,"seconds":timings,"top_threads":enabled,"accepted":exact}
    write_json(OUT/"cpu_threads_audit.json",proof);print("CPU最后层线程核验",proof,flush=True)


if __name__=="__main__":main()
