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

    def test_M4(self):
        for L in [4, 8, 16]:
            self.assertEqual(isUnitary(generateADSMCodes(4, "PSK", L)), True)

    def test_M2_O(self):
        for L in [2, 4, 8]:
            self.assertEqual(isUnitary(generateADSMDRCodes(2, "PSK", L, 2)), True)

        mind = getMinimumEuclideanDistance(generateADSMDRCodes(2, "PSK", 2, 2))
        self.assertAlmostEqual(mind, 1.0)
        mind = getMinimumEuclideanDistance(generateADSMDRCodes(2, "PSK", 4, 2))
        self.assertAlmostEqual(mind, 1.0)

    def test_M4_O(self):
        for O in [1, 2, 4]:
            for L in [2, 4, 8]:
                self.assertEqual(isUnitary(generateADSMDRCodes(4, "PSK", L, O)), True)

        mind = getMinimumEuclideanDistance(generateADSMDRCodes(4, "PSK", 2, 2))
        self.assertAlmostEqual(mind, 4.0)
        mind = getMinimumEuclideanDistance(generateADSMDRCodes(4, "PSK", 2, 4))
        self.assertAlmostEqual(mind, 1.0)
        mind = getMinimumEuclideanDistance(generateADSMDRCodes(4, "PSK", 4, 2))
        self.assertAlmostEqual(mind, 2.0)
        mind = getMinimumEuclideanDistance(generateADSMDRCodes(4, "PSK", 4, 4))
        self.assertAlmostEqual(mind, 1.0)


if __name__ == '__main__':
    unittest.main()
