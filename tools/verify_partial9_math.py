"""CPU数学/冻结边界测试；无需CUDA，不读取课程标签作为测试真值。"""
import unittest
import numpy as np
import torch
from task2.partial9 import FOLDER,load_top
from task2.finetune_protocol import forward_noise_loss
from task2.partial9_protocol import teacher_posterior


class PartialMathTests(unittest.TestCase):
    def test_residual_initialization_keeps_parent_and_gradient(self):
        original=torch.tensor([[2.,-1.,0.,.5,1.],[-.3,.2,1.,0.,-.2]],dtype=torch.float64)
        parent=torch.tensor([[.1,1.,.3,-.5,2.],[1.,.2,-1.,.8,.1]],dtype=torch.float64)
        student=original.clone().requires_grad_();effective=parent+student-original
        torch.testing.assert_close(torch.softmax(effective,1),torch.softmax(parent,1),atol=1e-15,rtol=1e-15)
        forward_noise_loss(torch,effective,torch.tensor([4,0]),.25).backward()
        direct=parent.clone().requires_grad_();forward_noise_loss(torch,direct,torch.tensor([4,0]),.25).backward()
        torch.testing.assert_close(student.grad,direct.grad,atol=1e-7,rtol=1e-6)

    def test_cached_cls_head_has_same_gradient_as_frozen_full_block(self):
        sample=np.load(FOLDER/"audit_batch.npz");a=load_top().train();b=load_top().train()
        for model in [a,b]:model.block.requires_grad_(False);model.block.eval()
        hidden=torch.from_numpy(sample["hidden"][:2]);mask=torch.from_numpy(sample["mask"][:2]).bool()
        torch.manual_seed(4321);logits,embedding=a(hidden,mask)
        forward_noise_loss(torch,logits,torch.tensor([0,4]),.25).backward()
        torch.manual_seed(4321)
        other=b.classifier(b.dropout(b.pooler(embedding.detach()[:,None,:])))
        forward_noise_loss(torch,other,torch.tensor([0,4]),.25).backward()
        torch.testing.assert_close(other,logits,atol=1e-6,rtol=1e-6)
        ga={k:p.grad for k,p in a.named_parameters() if p.requires_grad}
        for key,p in b.named_parameters():
            if p.requires_grad:torch.testing.assert_close(p.grad,ga[key],atol=1e-6,rtol=1e-5)

    def test_teacher_posterior_matches_bayes_enumeration(self):
        prior=np.array([[.6,.1,.1,.1,.1],[.15,.3,.2,.25,.1]])
        observed=np.array([4,1]);epsilon=.25;expected=[]
        for p,y in zip(prior,observed):
            # 独立枚举潜在标签和“保持/随机替换”两条生成路径，再按观测条件归一化。
            joint=[]
            for latent in range(5):
                mass=p[latent]*epsilon/5
                if latent==y:mass+=p[latent]*(1-epsilon)
                joint.append(mass)
            expected.append(np.array(joint)/sum(joint))
        np.testing.assert_allclose(teacher_posterior(prior,observed,epsilon),expected,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(teacher_posterior(prior,observed,0),np.eye(5)[observed],atol=0)

    def test_posterior_cross_entropy_gradient_and_hard_example(self):
        raw=np.array([[-12.,0.,0.,0.,12.]],np.float32)
        target=teacher_posterior(np.array([[.9,.025,.025,.025,.025]]),np.array([0]),.25)
        value=torch.tensor(raw,requires_grad=True)
        loss=-(torch.tensor(target)*torch.nn.functional.log_softmax(value,dim=-1)).sum()
        loss.backward();q=np.exp(raw-raw.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
        np.testing.assert_allclose(value.grad.numpy(),q-target,atol=1e-7,rtol=1e-6)
        other=torch.tensor(raw,requires_grad=True)
        forward_noise_loss(torch,other,torch.tensor([0]),.25).backward()
        self.assertLess(abs(float(other.grad[0,0])),1e-7)
        self.assertGreater(abs(float(value.grad[0,0])),.9)

    def test_noise_gradient_matches_independent_closed_form(self):
        raw=np.array([[2.,-.5,.3,-1.,1.],[.2,.6,1.1,-.8,-.2]],np.float32)
        y=np.array([0,3]);epsilon=.25165131324376977
        value=torch.tensor(raw,requires_grad=True)
        loss=forward_noise_loss(torch,value,torch.tensor(y),epsilon);loss.backward()
        q=np.exp(raw-raw.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
        qy=q[np.arange(2),y];observed=(1-epsilon)*qy+epsilon/5
        expected=q.copy();expected[np.arange(2),y]-=1
        expected*=(((1-epsilon)*qy/observed)/2)[:,None]
        np.testing.assert_allclose(value.grad.numpy(),expected,atol=1e-7,rtol=1e-6)
        self.assertAlmostEqual(float(loss),float(-np.log(observed).mean()),places=6)

    def test_frozen_block_and_relative_constant_do_not_update(self):
        torch.manual_seed(5434);sample=np.load(FOLDER/"audit_batch.npz")
        model=load_top().train();model.block.requires_grad_(False)
        before={k:v.clone() for k,v in model.block.state_dict().items()};relative=model.relative.clone()
        old_classifier=model.classifier.weight.detach().clone()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        logits,_=model(torch.from_numpy(sample["hidden"][:2]),torch.from_numpy(sample["mask"][:2]).bool())
        loss=forward_noise_loss(torch,logits,torch.tensor([0,4]),.25);loss.backward();optimizer.step()
        for k,v in model.block.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        torch.testing.assert_close(relative,model.relative,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.block.parameters()))
        self.assertFalse(torch.equal(old_classifier,model.classifier.weight))

    def test_unfrozen_encoder_block_receives_gradient(self):
        torch.manual_seed(5434);sample=np.load(FOLDER/"audit_batch.npz");model=load_top().train()
        logits,_=model(torch.from_numpy(sample["hidden"][:2]),torch.from_numpy(sample["mask"][:2]).bool())
        forward_noise_loss(torch,logits,torch.tensor([0,4]),.25).backward()
        gradient=model.block.attention.self.query_proj.weight.grad
        self.assertIsNotNone(gradient);self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.norm()),0)
        self.assertIsNone(model.relative.grad)


if __name__=="__main__":
    torch.set_num_threads(1);unittest.main()
