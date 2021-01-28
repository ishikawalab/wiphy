# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt
import os
import unittest

import numpy as np

import numpy as np
import wiphy.util.plot as me


class Test(unittest.TestCase):

    def test_getXCorrespondingToY(self):
        a = me.getXCorrespondingToY(np.array([0, 1]), np.array([0, 1]), 0.5)
        np.testing.assert_almost_equal(a, np.array(0.5))

    def test_getYCorrespondingToX(self):
        a = me.getYCorrespondingToX(np.array([0, 1]), np.array([0, 1]), 0.5)
        np.testing.assert_almost_equal(a, np.array(0.5))


if __name__ == '__main__':
    unittest.main()
