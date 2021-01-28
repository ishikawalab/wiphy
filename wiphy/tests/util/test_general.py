# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt
import os
import unittest

import numpy as np

import numpy as np
import wiphy.util.general as me
import wiphy.code.modulator as mod
import wiphy.code.im as im
import wiphy.code.duc as duc


class Test(unittest.TestCase):

    def test_getGrayIndixes(self):
        self.assertEqual(me.getGrayIndixes(2), [0, 1, 3, 2])
        self.assertEqual(me.getGrayIndixes(4), [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8])

    def test_frodiff(self):
        fro = me.frodiff(np.array([1, 1j, 0, 0]), np.array([1, 0, 1j, 0]))
        self.assertAlmostEqual(fro, 2.0, msg="The Frobenius norm calculation is wrong")
        fro = me.frodiff(me.randn_c(int(1e6)), me.randn_c(int(1e6)))
        self.assertAlmostEqual(fro / 2e6, 1.0, places=2)

    def test_matmulb(self):
        H = np.array([[1, 1.j], [-1.j, -1]])
        codes = im.generateIMCodes("opt", 2, 1, 2, "PSK", 1, 1)
        ret = me.matmulb(H, codes)
        np.testing.assert_almost_equal(ret, np.matmul(H, codes))

    def test_getEuclideanDistances(self):
        codes = mod.generatePSKSymbols(4).reshape(4, 1, 1)
        ret = me.asnumpy(me.getEuclideanDistances(np.array(codes)))
        np.testing.assert_almost_equal(ret, [2., 2., 4., 4., 2., 2.])
        #
        codes = im.generateIMCodes("opt", 2, 1, 2, "PSK", 1, 1)
        ret = me.asnumpy(me.getEuclideanDistances(np.array(codes)))
        np.testing.assert_almost_equal(ret, [2.])
        #
        codes = im.generateIMCodes("opt", 4, 2, 4, "PSK", 1, 1)
        ret = me.asnumpy(me.getEuclideanDistances(np.array(codes)))
        np.testing.assert_almost_equal(ret, [1., 1., 2., 2., 1., 1.])
        #
        codes = duc.generateDUCCodes(2, 2)
        ret = me.asnumpy(me.getEuclideanDistances(np.array(codes)))
        np.testing.assert_almost_equal(ret, [16.])

    def test_getMinimumEuclideanDistance(self):
        codes = mod.generatePSKSymbols(4).reshape(4, 1, 1)
        med = me.getMinimumEuclideanDistance(np.array(codes))
        self.assertAlmostEqual(med, 2.0)

        codes = mod.generateStarQAMSymbols(16).reshape(16, 1, 1)
        med = me.getMinimumEuclideanDistance(np.array(codes))
        self.assertAlmostEqual(med, 0.2343145750507619)

        codes = im.generateIMCodes("opt", 4, 2, 4, "PSK", 4, 1)
        med = me.getMinimumEuclideanDistance(np.array(codes))
        self.assertAlmostEqual(med, 1.0)

        codes = im.generateIMCodes("opt", 8, 4, 64, "PSK", 2, 1)
        med = me.getMinimumEuclideanDistance(np.array(codes))
        self.assertAlmostEqual(med, 0.5)

    def test_getDFTMatrix(self):
        W = me.getDFTMatrix(4)
        np.testing.assert_almost_equal(W.dot(W.conj().T), np.eye(4, dtype=np.complex), decimal=3)
        W = me.getDFTMatrix(8)
        np.testing.assert_almost_equal(W.dot(W.conj().T), np.eye(8, dtype=np.complex), decimal=3)
        W = me.getDFTMatrix(16)
        np.testing.assert_almost_equal(W.dot(W.conj().T), np.eye(16, dtype=np.complex), decimal=3)

    def test_inv_dB(self):
        self.assertAlmostEqual(me.inv_dB(0.0), 1.0, msg="The implementation of inv_dB may be wrong.")

    def test_randn(self):
        ret = me.randn(int(1e6))
        meanPower = np.mean(np.power(np.abs(ret), 2))
        self.assertAlmostEqual(meanPower, 1.0, places=2, msg="The mean power of randn differs from 1.0")

    def test_randn_c(self):
        ret = me.randn_c(int(1e6))
        meanPower = np.mean(np.power(np.abs(ret), 2))
        self.assertAlmostEqual(meanPower, 1.0, places=2, msg="The mean power of randn_c differs from 1.0")

    def test_countErrorBits(self):
        self.assertEqual(me.countErrorBits(1, 2), 2)
        self.assertEqual(me.countErrorBits(1, 5), 1)

    def test_getXORtoErrorBitsArray(self):
        a = me.getXORtoErrorBitsArray(4)
        np.testing.assert_almost_equal(a, np.array([0, 1, 1, 2, 1]))
        a = me.getXORtoErrorBitsArray(16)
        np.testing.assert_almost_equal(a, np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1]))

    def test_getErrorBitsTable(self):
        t = me.getErrorBitsTable(4)
        np.testing.assert_almost_equal(t, np.array([[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]]))
        t = me.getErrorBitsTable(8)
        np.testing.assert_almost_equal(t, np.array(
            [[0, 1, 1, 2, 1, 2, 2, 3], [1, 0, 2, 1, 2, 1, 3, 2], [1, 2, 0, 1, 2, 3, 1, 2], [2, 1, 1, 0, 3, 2, 2, 1],
             [1, 2, 2, 3, 0, 1, 1, 2], [2, 1, 3, 2, 1, 0, 2, 1], [2, 3, 1, 2, 1, 2, 0, 1], [3, 2, 2, 1, 2, 1, 1, 0]]))

    def test_getRandomHermitianMatrix(self):
        np.set_printoptions(linewidth=np.inf)
        H = me.getRandomHermitianMatrix(4)
        np.testing.assert_almost_equal(H, H.conj().T)
        H = me.getRandomHermitianMatrix(8)
        np.testing.assert_almost_equal(H, H.conj().T)
        H = me.getRandomHermitianMatrix(16)
        np.testing.assert_almost_equal(H, H.conj().T)

    def test_CayleyTransform(self):
        U = me.CayleyTransform(me.asnumpy(me.getRandomHermitianMatrix(4)))
        np.testing.assert_almost_equal(me.asnumpy(U.dot(U.conj().T)), np.eye(4, dtype=np.complex))
        U = me.CayleyTransform(me.asnumpy(me.getRandomHermitianMatrix(8)))
        np.testing.assert_almost_equal(me.asnumpy(U.dot(U.conj().T)), np.eye(8, dtype=np.complex))
        U = me.CayleyTransform(me.asnumpy(me.getRandomHermitianMatrix(16)))
        np.testing.assert_almost_equal(me.asnumpy(U.dot(U.conj().T)), np.eye(16, dtype=np.complex))

    def test_kurtosis(self):
        self.assertAlmostEqual(me.kurtosis(me.randn(10000000)), 0.0, places=2)


if __name__ == '__main__':
    unittest.main()
