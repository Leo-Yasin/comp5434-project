"""两条合成标签的小批CUDA检查：在正式长训练前验证确定性反向图可执行。"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from task2.partial9 import FOLDER,OUT,load_top
from task2.finetune_protocol import forward_noise_loss
from task2.audit import write_json


def main():
    torch.set_num_threads(1);torch.manual_seed(5434);torch.cuda.manual_seed_all(5434)
    torch.use_deterministic_algorithms(True)
    sample=np.load(FOLDER/"audit_batch.npz");model=load_top().cuda().train()
    hidden=torch.from_numpy(sample["hidden"][:2,:32].copy()).cuda()
    mask=torch.from_numpy(sample["mask"][:2,:32].copy()).cuda().bool()
    previous=model.block.attention.self.query_proj.weight.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,foreach=False)
    optimizer.zero_grad(set_to_none=True);logits,_=model(hidden,mask)
    loss=forward_noise_loss(torch,logits,torch.tensor([0,4],device="cuda"),.25165131324376977)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    if torch.equal(previous,model.block.attention.self.query_proj.weight):raise RuntimeError("编码层未更新。")
    record={"passed":True,"deterministic_algorithms":True,"rows":2,"tokens":32,"course_labels_used":False,
        "loss":float(loss.detach()),"cuda_reserved_bytes":torch.cuda.max_memory_reserved()}
    write_json(OUT/"deterministic_gpu_smoke.json",record);print("确定性CUDA反向图检查",record,flush=True)


if __name__=="__main__":main()
