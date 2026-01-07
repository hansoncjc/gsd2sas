import sys
import types
import unittest

import numpy as np

# Stub gsd import so structurefactor can be imported without gsd installed.
gsd_mod = types.ModuleType("gsd")
gsd_mod.hoomd = types.ModuleType("gsd.hoomd")
sys.modules.setdefault("gsd", gsd_mod)
sys.modules.setdefault("gsd.hoomd", gsd_mod.hoomd)

sys.path.insert(0, "../gsd2sas")
from gsd2sas import structurefactor as sf


class StructureFactorTorchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.N = 128
        self.box = np.array([10.0, 12.0, 14.0])
        self.x = rng.random((self.N, 3)) * self.box
        self.types = rng.integers(0, 2, size=self.N)

    def test_compute_s_3d_shapes(self):
        s3, n_grid = sf.compute_s_3d(self.x, self.box, N_grid=8)
        self.assertEqual(s3.shape, tuple(n_grid))
        self.assertTrue(np.isfinite(s3).all())

    def test_compute_s_1d_counts(self):
        q, num, cnt = sf.compute_s_1d(self.x, self.box, N_grid=8)
        self.assertEqual(q.shape, num.shape)
        self.assertEqual(num.shape, cnt.shape)
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue(np.isfinite(num).all())
        self.assertTrue(np.isfinite(cnt).all())
        self.assertTrue((cnt >= 0).all())
        self.assertTrue(np.any(cnt > 0))
        self.assertTrue(np.all(np.diff(q) >= 0))

    def test_compute_partial_s_1d_counts(self):
        q, s11_sum, s22_sum, s12_sum, cnt = sf.compute_partial_s_1d(
            self.x, self.types, self.box, N_grid=8
        )
        self.assertEqual(q.shape, s11_sum.shape)
        self.assertEqual(s11_sum.shape, s22_sum.shape)
        self.assertEqual(s22_sum.shape, s12_sum.shape)
        self.assertEqual(s12_sum.shape, cnt.shape)
        self.assertTrue(np.isfinite(s11_sum).all())
        self.assertTrue(np.isfinite(s22_sum).all())
        self.assertTrue(np.isfinite(s12_sum).all())
        self.assertTrue(np.isfinite(cnt).all())
        self.assertTrue((cnt >= 0).all())
        self.assertTrue(np.any(cnt > 0))
        self.assertTrue(np.all(np.diff(q) >= 0))


if __name__ == "__main__":
    unittest.main()
