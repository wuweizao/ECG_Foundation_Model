import unittest

import numpy as np
import pandas as pd

from src.labels import CLASSES, map_labels, patient_subset, validate_folds
from src.metrics import compute_metrics, thresholds_from_validation
from src.plan import conditions


class ProtocolTests(unittest.TestCase):
    def test_patient_leakage_rejected(self):
        frame = pd.DataFrame(dict(ecg_id=[1,2], patient_id=[9,9], strat_fold=[1,9]))
        with self.assertRaises(ValueError):
            validate_folds(frame)

    def test_nested_whole_patient_sampling(self):
        frame = pd.DataFrame(dict(patient_id=np.repeat(np.arange(100), 3)))
        small, large = patient_subset(frame, .05, 42), patient_subset(frame, .1, 42)
        self.assertTrue(set(small.patient_id) <= set(large.patient_id))
        self.assertEqual(len(small), 15)
        self.assertTrue((small.groupby('patient_id').size() == 3).all())
        pd.testing.assert_frame_equal(small, patient_subset(frame, .05, 42))

    def test_multilabel_and_zero_likelihood(self):
        statements = pd.DataFrame(dict(diagnostic=[1,1,0], diagnostic_class=['MI','STTC',None]), index=['A','B','R'])
        database = pd.DataFrame(dict(scp_codes=["{'A': 0, 'B': 100}", "{'R': 100}"]))
        result = map_labels(database, statements)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[CLASSES].values.tolist(), [[0,1,1,0,0]])

    def test_metrics_and_thresholds(self):
        y = np.tile([[0,1,0,1,0], [1,0,1,0,1]], (5,1))
        p = y * .8 + .1
        t = thresholds_from_validation(y,p)
        m, _ = compute_metrics(y,p,t)
        self.assertEqual(m['macro_auroc'], 1.)
        self.assertEqual(m['macro_f1'], 1.)
        self.assertAlmostEqual(m['brier'], .01)
        self.assertAlmostEqual(m['macro_ece'], .1)

    def test_degenerate_class_not_hidden(self):
        y = np.zeros((3,5))
        metrics, _ = compute_metrics(y, np.full_like(y,.2), [.5]*5)
        self.assertIsNone(metrics['macro_auroc'])

    def test_experiment_counts(self):
        plan = conditions()
        self.assertEqual(sum(r['model'] != 'linear_probe' for r in plan), 26)
        self.assertEqual(sum(r['model'] == 'linear_probe' for r in plan), 7)


if __name__ == '__main__':
    unittest.main()
