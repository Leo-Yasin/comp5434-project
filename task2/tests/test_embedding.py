"""CLS任务头关键数学检查：分块正规方程、噪声梯度、MLP反向传播与标签隔离。"""
import unittest
import numpy as np
from task2.embedding_heads import (fit_ridge,fit_linear,ridge_predict,linear_logits,noise_loss_gradient,
    mlp_loss_gradients,fit_mlp,mlp_predict)


class EmbeddingTests(unittest.TestCase):
    def test_linear_probe_normalization_and_labels_are_fit_only(self):
        rng=np.random.default_rng(24);x=rng.normal(size=(25,3));offset=rng.normal(size=(25,5))
        y=rng.integers(1,6,25).astype(float);fit=np.arange(25)<18
        a,_=fit_linear(x,offset,y,fit,.25,5.,iterations=30)
        y[~fit]=np.nan
        b,_=fit_linear(x,offset,y,fit,.25,5.,iterations=30)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_allclose(a["center"],x[fit].mean(axis=0),atol=1e-12)
        np.testing.assert_allclose(a["scale"],x[fit].std(axis=0),atol=1e-12)

    def test_chunked_ridge_matches_independent_direct_solution_and_ignores_heldout(self):
        rng=np.random.default_rng(67);x=rng.normal(size=(30,4));p=rng.uniform(1,5,30)
        y=rng.integers(1,6,size=30).astype(float);fit=np.arange(30)<23
        state,_=fit_ridge(x,p,y,fit,.25,3.)
        center=x[fit].mean(axis=0);scale=x[fit].std(axis=0)
        design=np.column_stack([np.ones(fit.sum()),(x[fit]-center)/scale])
        expected=np.linalg.solve(design.T@design+np.diag([1.,3.,3.,3.,3.]),
            design.T@((y[fit]-.75)/.75-p[fit]))
        np.testing.assert_allclose(state["coefficient"],expected,atol=1e-10)
        y[~fit]=np.nan;changed,_=fit_ridge(x,p,y,fit,.25,3.)
        np.testing.assert_array_equal(state["coefficient"],changed["coefficient"])

    def test_noise_logit_gradient_finite_difference(self):
        rng=np.random.default_rng(18);z=rng.normal(size=(4,5));y=np.array([1.,3.,5.,2.])
        value,g=noise_loss_gradient(z,y,.25);numerical=np.empty_like(z)
        for index in np.ndindex(z.shape):
            plus=z.copy();minus=z.copy();plus[index]+=1e-6;minus[index]-=1e-6
            numerical[index]=(noise_loss_gradient(plus,y,.25)[0]-noise_loss_gradient(minus,y,.25)[0])/2e-6
        np.testing.assert_allclose(g,numerical,atol=1e-8,rtol=1e-6)

    def test_mlp_full_chain_rule_with_fixed_dropout_finite_difference(self):
        rng=np.random.default_rng(38);x=rng.normal(size=(5,3));offset=rng.normal(size=(5,5))
        p={"w1":rng.normal(scale=.2,size=(3,2)),"b1":rng.normal(scale=.1,size=2),
            "w2":rng.normal(scale=.2,size=(2,5)),"b2":rng.normal(scale=.1,size=5)}
        y=np.array([1.,2.,3.,4.,5.]);mask=(rng.random((5,2))>.2)/.8
        loss,gradient=mlp_loss_gradients(x,offset,y,p,.25,.03,mask)
        for name in p:
            expected=np.empty_like(p[name])
            for index in np.ndindex(p[name].shape):
                a={k:v.copy() for k,v in p.items()};b={k:v.copy() for k,v in p.items()}
                a[name][index]+=1e-6;b[name][index]-=1e-6
                expected[index]=(mlp_loss_gradients(x,offset,y,a,.25,.03,mask)[0]-
                    mlp_loss_gradients(x,offset,y,b,.25,.03,mask)[0])/2e-6
            np.testing.assert_allclose(gradient[name],expected,atol=1e-8,rtol=1e-6)

    def test_mlp_training_does_not_read_validation_labels_and_is_reproducible(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(20,4)).astype(np.float32)
        offset=rng.normal(size=(20,5)).astype(np.float32);y=rng.integers(1,6,20).astype(float)
        fit=np.arange(20)<15;anchor={"center":np.zeros(4),"scale":np.ones(4),"weights":np.zeros((5,5))}
        first,_=fit_mlp(x,offset,y,fit,.25,anchor,epochs=2,hidden=3)
        y[~fit]=np.nan
        second,_=fit_mlp(x,offset,y,fit,.25,anchor,epochs=2,hidden=3)
        for key in first[2]:np.testing.assert_array_equal(first[2][key],second[2][key])
        p=mlp_predict(x,offset,first[2])
        self.assertTrue(np.isfinite(p).all() and ((p>=1)&(p<=5)).all())


if __name__=="__main__":unittest.main()
