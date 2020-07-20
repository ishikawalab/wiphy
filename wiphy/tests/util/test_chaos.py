# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import unittest

import numpy as np
import wiphy.util.chaos as cu
from scipy.integrate import odeint


class Test(unittest.TestCase):

    def test_lorenz(self):
        ret = odeint(cu.lorenz, np.array([1, 1, 1]), np.arange(0.0, 1, 0.1))
        np.testing.assert_almost_equal(ret, np.array(
            [[1., 1., 1.], [2.1331076, 4.47142016, 1.11389893], [6.54252716, 13.73118586, 4.18019692],
             [16.68481408, 27.18349319, 26.20646026], [15.36620005, 1.11303875, 46.75784257],
             [1.19827367, -8.86719686, 32.45474018], [-4.8332133, -8.06128061, 26.67318893],
             [-7.04936139, -8.74748642, 24.89008105], [-8.6355095, -10.09563998, 25.64672547],
             [-9.69220088, -10.15778157, 28.03033952]]))

    def test_getLogisticMapSequenceOriginal(self):
        W = 500  # 40000
        x0 = 1e-64
        reto = cu.getLogisticMapSequenceOriginal(x0, W)
        for d in np.arange(-64, 0, 1):
            ret = cu.getLogisticMapSequenceOriginal(x0 + np.power(10., d), W)
            self.assertNotEqual(0., np.sum(np.square(np.abs(reto - ret))))


if __name__ == '__main__':
    unittest.main()
