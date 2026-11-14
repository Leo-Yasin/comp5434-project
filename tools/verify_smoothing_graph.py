"""共享常量经Identity别名用于另一MatMul时，缩放不得改变该分支。需英语ONNX环境。"""
import unittest
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
import onnxruntime as ort
from task2.english_smooth import smooth_graph


class SharedGraphTests(unittest.TestCase):
    def test_alias_branch_and_scaled_branch_match_independent_numpy(self):
        w=np.array([[1.,2.],[-3.,4.],[.2,-.5]],dtype=np.float32)
        nodes=[helper.make_node("Identity",["W"],["alias_W"]),
               helper.make_node("MatMul",["x","W"],["a"],name="/direct/MatMul"),
               helper.make_node("MatMul",["y","alias_W"],["b"],name="/alias/MatMul")]
        graph=helper.make_graph(nodes,"shared",[helper.make_tensor_value_info(k,TensorProto.FLOAT,[None,3]) for k in ["x","y"]],
            [helper.make_tensor_value_info(k,TensorProto.FLOAT,[None,2]) for k in ["a","b"]],[numpy_helper.from_array(w,name="W")])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",17)]);model.ir_version=10
        model,count=smooth_graph(model,{"direct":np.array([100.,.1,4.],dtype=np.float32)})
        self.assertEqual(count,1)
        np.testing.assert_array_equal(numpy_helper.to_array(next(t for t in model.graph.initializer if t.name=="W")),w)
        session=ort.InferenceSession(model.SerializeToString(),providers=["CPUExecutionProvider"])
        x=np.array([[100.,.2,-4.],[2.,-.7,5.]],dtype=np.float32);y=x/2
        a,b=session.run(None,{"x":x,"y":y})
        np.testing.assert_allclose(a,x@w,atol=1e-5,rtol=1e-6);np.testing.assert_array_equal(b,y@w)


if __name__=="__main__":unittest.main()
