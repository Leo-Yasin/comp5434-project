"""检查新算法的基本学习性质、折内分箱隔离及序列化。"""
import unittest
import numpy as np
from task1.boosting import HistogramBoosting
from task1.generative import BackgroundHistogram


class NewModelTests(unittest.TestCase):
    def test_boosting_learns_and_roundtrips(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(300,3))
        y=np.digitize(x[:,0],[-0.7,0,0.7])+1
        model=HistogramBoosting(rounds=40,depth=2,min_leaf=5).fit(x,y)
        p=model.predict_proba(x)
        self.assertGreater((p.argmax(1)+1==y).mean(),0.94)
        self.assertLess(model.loss_history[-1]["cross_entropy"],model.loss_history[0]["cross_entropy"])
        np.testing.assert_array_equal(p,HistogramBoosting.from_dict(model.to_dict()).predict_proba(x))
        cuts=[v.copy() for v in model.cuts];model.predict_proba(np.full((2,3),1e6))
        for a,b in zip(cuts,model.cuts):np.testing.assert_array_equal(a,b)

    def test_histogram_probability_finite_and_roundtrip(self):
        x=np.repeat(np.arange(4),20).reshape(-1,1);y=np.repeat(np.arange(1,5),20)
        for bg in [False,True]:
            model=BackgroundHistogram(background=bg).fit(x,y)
            p=model.predict_proba(x)
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p.sum(1),1)
            np.testing.assert_array_equal(p,BackgroundHistogram.from_dict(model.to_dict()).predict_proba(x))


if __name__=="__main__":unittest.main()
