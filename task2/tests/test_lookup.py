"""验证文本规范化、键构建无标签、无历史回退与残差组合的边界。"""
import unittest
import task2
import numpy as np
import pandas as pd
from task2.lookup_features import make_keys,KEYS
from task2.lookup_model import lookup,residual_design,predict_residual


class LookupTests(unittest.TestCase):
    def test_normalized_keys_do_not_use_rating(self):
        f=pd.DataFrame({"title":["Great Product"," great   product "],
                        "comment":["Very nice product indeed","very nice  product indeed"],
                        "user_id":["u","u"],"prod_id":["p","p"],"parent_prod_id":["q","q"],
                        "purchased":[True,True]})
        keys=make_keys(f)
        for name in KEYS:self.assertEqual(keys[name][0],keys[name][1])
        f["rating"]=[1,5]
        for key,value in make_keys(f).items():np.testing.assert_array_equal(value,keys[key])

    def test_unknown_and_invalid_keys_have_zero_support(self):
        state={"keys":np.array([10,20],dtype=np.uint64),"counts":np.array([2,3]),
               "sums":np.array([9.,6.]),"epsilon":np.array(.25)}
        mean,count=lookup(np.array([10,99,20],dtype=np.uint64),state,np.array([True,True,False]))
        np.testing.assert_array_equal(count,[2,0,0]);self.assertAlmostEqual(mean[0],5.)
        base=np.array([2.,3.,4.]);means=np.tile(mean[:,None],(1,len(KEYS)))
        counts=np.tile(count[:,None],(1,len(KEYS)))
        design=residual_design(base,means,counts)
        np.testing.assert_array_equal(design[1:,2:],0)
        np.testing.assert_array_equal(predict_residual(base,means,counts,np.zeros(design.shape[1])),base)

    def test_context_and_prediction_shape(self):
        base=np.array([2.,4.]);means=np.ones((2,len(KEYS)))*3;counts=np.ones_like(means)
        context=np.zeros((2,9));design=residual_design(base,means,counts,context)
        self.assertEqual(design.shape,(2,28))
        np.testing.assert_array_equal(predict_residual(base,means,counts,np.zeros(28),context),base)


if __name__=="__main__":unittest.main()
