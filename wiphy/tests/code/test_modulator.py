# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import unittest

import numpy as np
from wiphy.code.modulator import *
from wiphy.util.general import getMinimumEuclideanDistance


class Test(unittest.TestCase):

    def test_PSK(self):
        for L in 2 ** np.arange(1, 8, 1):
            symbols = generatePSKSymbols(L)
            meanNorm = np.mean(np.square(np.abs(symbols)))
            self.assertAlmostEqual(meanNorm, 1.0, msg="The mean power of PSK(" + str(L) + ") symbols differs from 1.0")
            med = getMinimumEuclideanDistance(symbols.reshape(L, 1, 1))
            self.assertGreater(med, 0, msg="The minimum Euclidean distance of PSK(" + str(L) + ") symbols is too small")

    def test_QAM(self):
        for L in 2 ** np.arange(2, 8, 2):
            symbols = generateQAMSymbols(L)
            meanNorm = np.mean(np.square(np.abs(symbols)))
            self.assertAlmostEqual(meanNorm, 1.0, msg="The mean power of QAM(" + str(L) + ") symbols differs from 1.0")
            med = getMinimumEuclideanDistance(symbols.reshape(L, 1, 1))
            self.assertGreater(med, 0, msg="The minimum Euclidean distance of QAM(" + str(L) + ") symbols is too small")

    def test_StarQAM(self):
        for L in 2 ** np.arange(1, 8, 1):
            symbols = generateStarQAMSymbols(L)
            meanNorm = np.mean(np.square(np.abs(symbols)))
            self.assertAlmostEqual(meanNorm, 1.0,
                                   msg="The mean power of StarQAM(" + str(L) + ") symbols differs from 1.0")
            med = getMinimumEuclideanDistance(symbols.reshape(L, 1, 1))
            self.assertGreater(med, 0,
                               msg="The minimum Euclidean distance of StarQAM(" + str(L) + ") symbols is too small")


if __name__ == '__main__':
    unittest.main()
