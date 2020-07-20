# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import unittest

import numpy as np
from wiphy.code.adsm import *
from wiphy.util.general import getMinimumEuclideanDistance, isUnitary


class Test(unittest.TestCase):
    def test_M2(self):
        for L in [2, 4, 8]:
            self.assertEqual(isUnitary(generateADSMCodes(2, "PSK", L)), True)

        mind = getMinimumEuclideanDistance(generateADSMCodes(2, "PSK", 2))
        self.assertAlmostEqual(mind, 4.0)
        mind = getMinimumEuclideanDistance(generateADSMCodes(2, "PSK", 2, 2.0 * np.pi / 4.0))
        self.assertAlmostEqual(mind, 2.0)

    def test_M4(self):
        for L in [4, 8, 16]:
            self.assertEqual(isUnitary(generateADSMCodes(4, "PSK", L)), True)


if __name__ == '__main__':
    unittest.main()
