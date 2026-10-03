import unittest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from src.phase3.register import allocation,subset
from src.phase3.train import thresholds,predict_bce


class PhaseThreeTests(unittest.TestCase):
    def test_exact_total_patient_budget(self):
        for f,expected in [(.01,(149,19)),(.05,(742,95)),(.1,(1483,191)),(.25,(3706,479)),(1.,(14823,1917))]:
            self.assertEqual(allocation(f,14823,1917),expected)
        with self.assertRaises(ValueError):
            allocation(.00001,14823,1917)

    def test_nested_whole_patient_prefix_without_label_redraw(self):
        frame=pd.DataFrame({'patient_id':np.repeat(np.arange(100),3),'label':0})
        small,large=subset(frame,5,42),subset(frame,20,42)
        self.assertEqual(len(small),15)
        self.assertEqual(small.patient_id.nunique(),5)
        self.assertTrue(set(small.patient_id)<=set(large.patient_id))

    def test_missing_validation_class_has_explicit_fixed_threshold(self):
        y=np.array([[0,1,0,1,0],[0,1,1,0,1],[0,1,0,1,1]])
        p=np.full((3,5),.4)
        values,fallback=thresholds(y,p)
        self.assertEqual(fallback,[True,True,False,False,False])
        self.assertEqual(values[:2],[.5,.5])
        self.assertEqual(values[2:],[.01,.01,.01])

    def test_validation_bce_weights_partial_batch_by_elements(self):
        x=torch.arange(15,dtype=torch.float32).reshape(3,5)/5
        y=torch.tensor([[0.,1.,0.,1.,0.],[1.,0.,1.,0.,1.],[1.,1.,1.,1.,1.]])
        loader=DataLoader(TensorDataset(x,y),batch_size=2)
        actual_y,p,loss=predict_bce(torch.nn.Identity(),loader,torch.device('cpu'),False)
        self.assertAlmostEqual(loss,float(torch.nn.functional.binary_cross_entropy_with_logits(x,y)),places=6)
        self.assertTrue(np.array_equal(actual_y,y.numpy()))
        self.assertTrue(np.array_equal(p,torch.sigmoid(x).numpy()))


if __name__=='__main__':
    unittest.main()
