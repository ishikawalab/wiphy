# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import unittest

from wiphy.code.tast import *
from wiphy.util.general import isUnitary, getMinimumEuclideanDistance


class Test(unittest.TestCase):

    def test_M2(self):
        codes = generateTASTCodes(2, 1, 2)
        self.assertGreater(getMinimumEuclideanDistance(codes), 0.0)
        self.assertEqual(isUnitary(codes), True)

        codes = generateTASTCodes(2, 2, 2)
        self.assertGreater(getMinimumEuclideanDistance(codes), 0.0)
        self.assertEqual(isUnitary(codes), True)

        codes = generateTASTCodes(2, 4, 2)
        self.assertGreater(getMinimumEuclideanDistance(codes), 0.0)
        self.assertEqual(isUnitary(codes), True)

        codes = generateTASTCodes(2, 2, 16)
        self.assertGreater(getMinimumEuclideanDistance(codes), 0.0)
        self.assertEqual(isUnitary(codes), True)

    def test_M4(self):
        codes = generateTASTCodes(4, 1, 2)
        self.assertGreater(getMinimumEuclideanDistance(codes), 0.0)
        self.assertEqual(isUnitary(codes), True)

        codes = generateTASTCodes(4, 2, 2)
        self.assertGreater(getMinimumEuclideanDistance(codes), 0.0)
        self.assertEqual(isUnitary(codes), True)

        codes = generateTASTCodes(4, 2, 4)
        self.assertGreater(getMinimumEuclideanDistance(codes), 0.0)
        self.assertEqual(isUnitary(codes), True)


if __name__ == '__main__':
    unittest.main()
