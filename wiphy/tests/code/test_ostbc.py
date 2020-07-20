# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import unittest

from wiphy.code.ostbc import *
from wiphy.util.general import isUnitary


class Test(unittest.TestCase):

    def test_M2(self):
        for L in [2, 4, 8, 16]:
            self.assertEqual(isUnitary(generateOSTBCodes(2, "PSK", L)), True)

    def test_M4(self):
        for L in [2, 4, 8, 16]:
            self.assertEqual(isUnitary(generateOSTBCodes(4, "PSK", L, nsymbols=2)), True)
            self.assertEqual(isUnitary(generateOSTBCodes(4, "PSK", L, nsymbols=3)), True)

    def test_M16(self):
        self.assertEqual(isUnitary(generateOSTBCodes(16, "PSK", 2)), True)


if __name__ == '__main__':
    unittest.main()
