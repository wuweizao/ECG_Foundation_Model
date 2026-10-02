import unittest
import torch
from src.phase2.models import build, tensor_hash
from src.phase2.suite import choose
from src.phase2.baseline_head import create_head1d
from src.utils import seed_all


class PhaseTwoTests(unittest.TestCase):
    def test_selection_ignores_failed_and_breaks_ties_by_lr(self):
        rows=[dict(spec=dict(lr=.001),score=.9),dict(spec=dict(lr=.0001),score=.9),
              dict(spec=dict(lr=.00001),score=None)]
        self.assertEqual(choose(rows)['spec']['lr'],.0001)
        with self.assertRaises(RuntimeError):
            choose([rows[-1]])

    def test_default_head_pool_order(self):
        head=create_head1d(2,5).eval()
        x=torch.tensor([[[1.,3.],[4.,2.]],[[0.,2.],[6.,8.]]])
        z=head[1](head[0](x))
        self.assertTrue(torch.equal(z,torch.tensor([[3.,4.,2.,3.],[2.,8.,1.,7.]])))

    def test_random_probe_freezes_encoder_and_preserves_head_initialization(self):
        seed_all(42)
        a=build(dict(family='scratch'),dict(checkpoint=None))
        head=tensor_hash(a.dense.state_dict())
        del a
        seed_all(42)
        b=build(dict(family='random_probe'),dict(checkpoint=None))
        self.assertEqual(head,tensor_hash(b.dense.state_dict()))
        self.assertTrue(all(p.requires_grad==n.startswith('dense.') for n,p in b.named_parameters()))

    def test_baseline_accepts_complete_12lead_record_and_backpropagates(self):
        seed_all(42)
        model=build(dict(family='xresnet1d101'),{})
        out=model(torch.randn(2,12,5000))
        self.assertEqual(tuple(out.shape),(2,5))
        out.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__=='__main__':
    unittest.main()
