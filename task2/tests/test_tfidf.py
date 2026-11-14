"""TF-IDF的训练/预测共用变换验证：独立手算、分块范数与数值列隔离。"""
import unittest
import numpy as np
from scipy import sparse
from task2.tfidf_sparse import transform,WORD_DIM
from task2.features import TEXT_DIM,TITLE_DIM


class TfidfTests(unittest.TestCase):
    def test_block_norm_and_numeric_preservation(self):
        indices=np.array([0,1,TITLE_DIM,TITLE_DIM+1,WORD_DIM,0,TITLE_DIM,WORD_DIM+2])
        values=np.array([1.,-2.,3.,4.,.7,2.,-3.,.2],dtype=np.float32)
        x=sparse.csr_matrix((values,indices,np.array([0,5,8])),shape=(2,TEXT_DIM))
        idf=np.ones(WORD_DIM,dtype=np.float32);idf[1]=2.;idf[TITLE_DIM+1]=.5
        result=transform(x,idf)
        np.testing.assert_allclose(result[0,:2].toarray(),[[1.5/np.sqrt(17),-6/np.sqrt(17)]],atol=1e-7)
        np.testing.assert_allclose(np.asarray(result[:,:TITLE_DIM].multiply(result[:,:TITLE_DIM]).sum(axis=1)).ravel(),2.25,atol=1e-6)
        np.testing.assert_allclose(np.asarray(result[:,TITLE_DIM:WORD_DIM].multiply(result[:,TITLE_DIM:WORD_DIM]).sum(axis=1)).ravel(),1.,atol=1e-6)
        np.testing.assert_array_equal(x[:,WORD_DIM:].toarray(),result[:,WORD_DIM:].toarray())
        np.testing.assert_array_equal(x.data,values)


if __name__=="__main__":unittest.main()
