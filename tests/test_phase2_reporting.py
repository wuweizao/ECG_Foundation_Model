import json
import unittest
from pathlib import Path
import numpy as np
from src.metrics import compute_metrics
from src.phase2.report import independent_metrics
from src.phase2.evaluate import completed_selection
from src.phase2.register import verify_phase1


class PhaseTwoReportingTests(unittest.TestCase):
    def test_independent_metrics_handle_tied_scores(self):
        rng=np.random.default_rng(14)
        y=rng.integers(0,2,size=(80,5))
        p=np.round(rng.uniform(size=(80,5)),1)
        threshold=[.3,.4,.5,.6,.7]
        metrics,_=compute_metrics(y,p,threshold)
        independent=independent_metrics(y,p,threshold)
        for key,value in independent.items():
            self.assertAlmostEqual(value,metrics[key],places=12)

    def test_phase_one_seal_remains_valid(self):
        verify_phase1()

    def test_incomplete_queue_cannot_select_for_test(self):
        if not Path('results/phase2/training_manifest.json').exists():
            with self.assertRaises(RuntimeError):
                completed_selection()


if __name__=='__main__':
    unittest.main()
