# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import unittest

import numpy as np
from wiphy.code.anm import *


class Test(unittest.TestCase):

    def test_all(self):
        codes = generateANMCodes(2, "PSK", 2)
        np.testing.assert_almost_equal(codes, np.array(
            [[[1.], [0.]], [[-1.], [0.]], [[0.70710678], [0.70710678]], [[-0.70710678], [-0.70710678]]]))

        for M in [2, 4, 8]:
            for L in [2, 4, 8]:
                codes = generateANMCodes(M, "PSK", L)
                self.assertAlmostEqual(np.mean(codes), 0.0)
                self.assertAlmostEqual(np.mean(np.sum(np.square(np.abs(codes)), axis=1)), 1.0)

            for L in [4, 16, 64]:
                codes = generateANMCodes(M, "QAM", L)
                self.assertAlmostEqual(np.mean(codes), 0.0)
                self.assertAlmostEqual(np.mean(np.sum(np.square(np.abs(codes)), axis=1)), 1.0)


if __name__ == '__main__':
    unittest.main()
