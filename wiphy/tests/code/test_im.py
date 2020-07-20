# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import unittest

import numpy as np
from wiphy.code.im import *
from wiphy.util.general import getMinimumEuclideanDistance


class Test(unittest.TestCase):

    def test_M2(self):
        codes = generateIMCodes("opt", 2, 1, 2, "PSK", 1, 1)
        np.testing.assert_almost_equal(codes, np.array([[[1.], [0.]], [[0.], [1.]]]))
        codes = generateIMCodes("dic", 4, 1, 4, "PSK", 1, 1)
        np.testing.assert_almost_equal(codes, np.array(
            [[[1.], [0.], [0.], [0.]], [[0.], [1.], [0.], [0.]], [[0.], [0.], [1.], [0.]], [[0.], [0.], [0.], [1.]]]))
        codes = generateIMCodes("wen", 4, 2, 4, "PSK", 1, 1)
        np.testing.assert_almost_equal(codes, np.array(
            [[[0.70710678], [0.70710678], [0.], [0.]], [[0.], [0.70710678], [0.70710678], [0.]],
             [[0.], [0.], [0.70710678], [0.70710678]], [[0.70710678], [0.], [0.], [0.70710678]]]))
        codes = generateIMCodes("dic", 8, 4, 8, "PSK", 2, 1)
        self.assertAlmostEqual(getMinimumEuclideanDistance(codes), 0.5)
        codes = generateIMCodes("opt", 8, 4, 8, "PSK", 2, 1)
        self.assertAlmostEqual(getMinimumEuclideanDistance(codes), 1.0)


if __name__ == '__main__':
    unittest.main()
