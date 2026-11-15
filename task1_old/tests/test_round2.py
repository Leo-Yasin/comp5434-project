"""检查多模式密度、随机阈值树与支持区间专家的数值/隔离性质。"""
import unittest
import numpy as np
import pandas as pd
from task1.data import NUMERIC,CATEGORICAL
from task1.mixture import ConditionalMixture
from task1.random_threshold import RandomThresholdForest
from task1.models import model_from_dict
from task1.core_experiment import fit_support,support_mask


class Round2Tests(unittest.TestCase):
    def test_capacity_assignment_matches_exhaustive_optimum(self):
        """在 3^6 个小型分配中穷举，独立验证配额算法没有停在次优解。"""
        import itertools
        from task1.prior_assignment import constrained_assignment,prior_quotas
        rng=np.random.default_rng(18)
        for _ in range(3):
            p=rng.random((6,3));p/=p.sum(1,keepdims=True);quota=prior_quotas(6,[1,1,1])
            labels=constrained_assignment(p,quota)
            optimum=-np.inf
            for candidate in itertools.product(range(3),repeat=6):
                if np.array_equal(np.bincount(candidate,minlength=3),quota):
                    optimum=max(optimum,p[np.arange(6),candidate].sum())
            self.assertAlmostEqual(p[np.arange(6),labels].sum(),optimum)
            np.testing.assert_array_equal(np.bincount(labels,minlength=3),quota)

    def test_mixture_learns_and_roundtrips(self):
        rng=np.random.default_rng(12);labels=np.repeat(np.arange(1,5),40)
        frame=pd.DataFrame({col:labels*3+rng.normal(scale=.3,size=len(labels)) for col in NUMERIC})
        for col in CATEGORICAL:frame[col]=np.where(labels<=2,"a","b")
        model=ConditionalMixture(components=3,iterations=30).fit(frame,labels)
        p=model.predict_proba(frame)
        self.assertGreater((p.argmax(1)+1==labels).mean(),.95)
        np.testing.assert_allclose(p.sum(1),1)
        np.testing.assert_array_equal(p,ConditionalMixture.from_dict(model.to_dict()).predict_proba(frame))

    def test_constant_mixture_is_finite(self):
        y=np.repeat(np.arange(1,5),10);f=pd.DataFrame({col:np.ones(40) for col in NUMERIC})
        for col in CATEGORICAL:f[col]="same"
        p=ConditionalMixture(components=2,iterations=15).fit(f,y).predict_proba(f)
        self.assertTrue(np.isfinite(p).all());np.testing.assert_allclose(p.sum(1),1)

    def test_random_threshold_forest_learns_and_roundtrips(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(300,2));y=np.where(x[:,0]>0,1,2)
        model=RandomThresholdForest(trees=20,max_features=2,min_leaf=2).fit(x,y)
        p=model.predict_proba(x);self.assertGreater((p.argmax(1)+1==y).mean(),.95)
        np.testing.assert_array_equal(p,model_from_dict(model.to_dict()).predict_proba(x))

    def test_support_mask_reuses_training_parameters(self):
        rng=np.random.default_rng(14);y=np.repeat(np.arange(1,5),30)
        f=pd.DataFrame({"Atmospheric Pressure":980+y*10+rng.uniform(size=len(y))})
        for col in CATEGORICAL:f[col]="same"
        support=fit_support(f,y,.35)
        import json
        before=json.dumps(support)
        unknown=f.iloc[:2].copy();unknown["Atmospheric Pressure"]=2000
        self.assertFalse(support_mask(unknown,support).any())
        self.assertEqual(before,json.dumps(support))


if __name__=="__main__":unittest.main()
