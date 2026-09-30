import unittest
import numpy as np

from src.dataset import LEADS, preprocess
from src.robustness import corrupt
from src.bootstrap import patient_indices


class SignalTests(unittest.TestCase):
    def test_lead_order_and_normalization(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(5000,12))
        base = preprocess(x, LEADS, 500)
        perm = rng.permutation(12)
        shuffled = preprocess(x[:,perm], [LEADS[i] for i in perm], 500)
        np.testing.assert_array_equal(base, shuffled)
        self.assertAlmostEqual(float(base.mean()), 0., places=6)
        self.assertAlmostEqual(float(base.std()), 1., places=5)

    def test_invalid_waveform(self):
        with self.assertRaises(ValueError):
            preprocess(np.zeros((1000,12)), LEADS, 100)
        x = np.zeros((5000,12))
        x[0,0] = np.nan
        with self.assertRaises(ValueError):
            preprocess(x, LEADS, 500)

    def test_noise_snr_and_pairing(self):
        x = np.random.default_rng(1).normal(size=(12,5000)).astype(np.float32)
        noisy = corrupt(x,'gaussian',10,4)
        snr = 10*np.log10(np.mean(x*x)/np.mean((noisy-x)**2))
        self.assertAlmostEqual(snr,10.,places=4)
        np.testing.assert_array_equal(noisy,corrupt(x,'gaussian',10,4))
        dropped = corrupt(x,'dropout',3,4)
        self.assertEqual(int((np.abs(dropped).sum(axis=1)==0).sum()),3)

    def test_bootstrap_retains_patient_clusters(self):
        patient = np.array([1,1,2,3,3,3])
        idx = patient_indices(patient,np.random.default_rng(7))
        counts = np.bincount(idx,minlength=len(patient))
        self.assertEqual(counts[0],counts[1])
        self.assertEqual(counts[3],counts[4])
        self.assertEqual(counts[4],counts[5])


if __name__ == '__main__':
    unittest.main()
