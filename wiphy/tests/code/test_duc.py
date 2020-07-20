# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import unittest

from wiphy.code.duc import *
from wiphy.util.general import isUnitary


class Test(unittest.TestCase):

    def test_M2(self):
        for L in [2, 4, 16, 256]:
            self.assertEqual(isUnitary(generateDUCCodes(2, L)), True)

    def test_M4(self):
        for L in [4, 16, 256]:
            self.assertEqual(isUnitary(generateDUCCodes(4, L)), True)


if __name__ == '__main__':
    unittest.main()
