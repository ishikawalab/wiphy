# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import unittest

import numpy as np
import wiphy.util.im as imu


class IMUtilTest(unittest.TestCase):
    def test_convertIndsToVector(self):
        ret = imu.convertIndsToVector([[0, 1], [0, 2]], M=4)
        np.testing.assert_array_almost_equal(ret, [np.array([[1], [1], [0], [0]]), np.array([[1], [0], [1], [0]])])

    def test_convertIndsToMatrix(self):
        ret = imu.convertIndsToMatrix([[0, 1], [0, 2]], M=4)
        np.testing.assert_array_almost_equal(ret, [np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.]]),
                                                   np.array([[1., 0.], [0., 0.], [0., 1.], [0., 0.]])])

    def test_convertIndsToIndsDec(self):
        ret = imu.convertIndsToIndsDec([[0, 1], [0, 2]], M=4)
        self.assertEqual(ret, [3, 5])

    def test_convertIndsDecToInds(self):
        ret = imu.convertIndsDecToInds(indsdecs=[3, 5, 10, 12], M=4)
        self.assertEqual(ret, [[0, 1], [0, 2], [1, 3], [2, 3]])

    def test_getIndexes(self):
        self.assertEqual(imu.getIndexes("opt", 4, 2, 4), [[0, 1], [0, 3], [1, 2], [2, 3]])
        self.assertEqual(imu.getIndexes("dic", 4, 2, 4), [[0, 1], [0, 2], [0, 3], [1, 2]])
        self.assertEqual(imu.getIndexes("mes", 4, 2, 4), [[2, 3], [1, 3], [1, 2], [0, 3]])
        self.assertEqual(imu.getIndexes("wen", 4, 2, 4), [[0, 1], [1, 2], [2, 3], [0, 3]])

    def test_getGoodDecsTableSmallMemory(self):
        self.assertEqual(imu.getGoodDecsTableSmallMemory(4, 2), {4: [3, 12]})
        self.assertEqual(imu.getGoodDecsTableSmallMemory(8, 4), {4: [15, 51, 195, 85, 165, 153, 105, 150, 102, 90, 170, 60, 204, 240], 6: [15, 113], 8: [15, 240]})

    def test_writeDecTable_readDecTable(self):
        imu.writeDecTable(4, 2)
        decTable = imu.readDecTable(4, 2)
        self.assertEqual(decTable, {4: [3, 12]})
        imu.writeDecTable(8, 4)
        decTable = imu.readDecTable(8, 4)
        self.assertEqual(decTable, {4: [15, 51, 195, 85, 165, 153, 105, 150, 102, 90, 170, 60, 204, 240], 6: [15, 113], 8: [15, 240]})

    def test_getAllIndsBasedOnDecFile(self):
        self.assertEqual(imu.getAllIndsBasedOnDecFile(4, 2, 2), [[0, 1], [2, 3]])
        self.assertEqual(imu.getAllIndsBasedOnDecFile(4, 2, 4), [])
        self.assertEqual(imu.getAllIndsBasedOnDecFile(8, 4, 8), [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 6, 7], [0, 2, 4, 6], [0, 2, 5, 7], [0, 3, 4, 7], [0, 3, 5, 6], [1, 2, 4, 7], [1, 2, 5, 6], [1, 3, 4, 6], [1, 3, 5, 7], [2, 3, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7]])

    def test_getGoodIndsBasedOnDecFile(self):
        np.testing.assert_array_equal(imu.getGoodIndsBasedOnDecFile(4, 2, 4), [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        np.testing.assert_array_equal(imu.getGoodIndsBasedOnDecFile(8, 4, 2), [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_getProbabilityOfActivation(self):
        ret = imu.getProbabilityOfActivation([[0], [1], [2], [3]], 4)
        np.testing.assert_array_almost_equal(ret, np.ones(4) / 4)

        ret = imu.getProbabilityOfActivation([[0, 1], [0, 3], [1, 2], [2, 3]], 4)
        np.testing.assert_array_almost_equal(ret, np.ones(4) / 2)

    def test_getHammingDistance(self):
        self.assertEqual(imu.getHammingDistance([0, 1], [1, 0]), 2)
        self.assertEqual(imu.getHammingDistance([1, 1, 0, 0], [0, 0, 1, 1]), 4)

    def test_getMinimumHammingDistance(self):
        minh = imu.getMinimumHammingDistance(imu.getIndexes("opt", 16, 8, 16), 16)
        self.assertEqual(minh, 8)
        minh = imu.getMinimumHammingDistance(imu.getIndexes("dic", 16, 8, 16), 16)
        self.assertEqual(minh, 2)

    def test_getSumHamming(self):
        ret = imu.getSumHamming(inds=[[0, 1], [2, 3], [1, 2], [0, 3]], M=4)
        self.assertEqual(ret, 16)

    def test_checkConflict(self):
        self.assertFalse(imu.checkConflict([[0, 1], [0, 2]]))
        self.assertTrue(imu.checkConflict([[0, 1], [0, 1]]))

    def test_getDictionaryIndexesList(self):
        ret = imu.getDictionaryIndexesList(4, 2, 4)
        self.assertTrue(ret == [[0, 1], [0, 2], [0, 3], [1, 2]])

    def test_wen2016EquiprobableSubcarrierActivation(self):
        ret = imu.wen2016EquiprobableSubcarrierActivation(M=2, K=1)
        np.testing.assert_array_equal(ret, [[0], [1]])
        ret = imu.wen2016EquiprobableSubcarrierActivation(M=4, K=1)
        np.testing.assert_array_equal(ret, [[0], [1], [2], [3]])
        ret = imu.wen2016EquiprobableSubcarrierActivation(M=4, K=2)
        np.testing.assert_array_equal(ret, [[0, 1], [1, 2], [2, 3], [0, 3]])

    def test_getIMParameters(self):
        np.testing.assert_array_equal(imu.getIMParameters(4, 2), [[4, 2, 2], [4, 2, 4]])


if __name__ == '__main__':
    unittest.main()
