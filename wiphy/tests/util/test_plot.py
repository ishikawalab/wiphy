# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt
import os
import unittest

import numpy as np

if os.getenv("USECUPY") == "1":
    import cupy as xp
else:
    import numpy as xp
import wiphy.util.plot as me


class Test(unittest.TestCase):

    def test_getXCorrespondingToY(self):
        a = me.getXCorrespondingToY(xp.array([0, 1]), xp.array([0, 1]), 0.5)
        np.testing.assert_almost_equal(a, xp.array(0.5))

    def test_getYCorrespondingToX(self):
        a = me.getYCorrespondingToX(xp.array([0, 1]), xp.array([0, 1]), 0.5)
        np.testing.assert_almost_equal(a, xp.array(0.5))


if __name__ == '__main__':
    unittest.main()
