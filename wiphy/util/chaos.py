# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

__all__ = ['lorenz', 'RungeKutta', 'odeintRungeKutta', 'logisticMap', 'logisticMapClosedForm', 'getLogisticMapSequence',
           'getLogisticMapSequenceOriginal', 'getUniformLogisticMapSequenceOriginal',
           'getSecondChebyshevPolynomialSequence']

import numpy as np
from numba import jit


def lorenz(p, t, rho=28.0, sigma=10.0, beta=8.0 / 3.0):
    return np.array([sigma * (p[1] - p[0]), p[0] * (rho - p[2]) - p[1], p[0] * p[1] - beta * p[2]])


def RungeKutta(f, p, t, h):
    k1 = f(p, t)
    k2 = f(p + h / 2.0 * k1, t + h / 2.0)
    k3 = f(p + h / 2.0 * k2, t + h / 2.0)
    k4 = f(p + h * k3, t + h)
    return p + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@jit
def odeintRungeKutta(f, initialp, ts):
    ret = np.zeros((len(ts), 3))
    ret[0, :] = initialp
    for i in range(len(ts) - 1):
        ret[i + 1, :] = RungeKutta(f, ret[i, :], ts[i], ts[i + 1] - ts[i])
    return ret


@jit
def logisticMap(xn, a=4.0):
    return a * xn * (1.0 - xn)


# with a = 4.0
def logisticMapClosedForm(x0, i):
    return np.square(np.sin(np.exp2(i) * np.arcsin(np.sqrt(x0))))


@jit
def getLogisticMapSequence(x0, size):
    ret = np.zeros(size)
    ret[0] = x0
    ret[1:size] = logisticMapClosedForm(x0, np.arange(1, size))
    # asqx0 = np.arcsin(np.sqrt(x0))
    # for i in range(1, len(ret)):
    #     ret[i] = np.square(np.sin(2**i * asqx0))
    #     2 ** (i + log2(asqx0))
    return ret


# x0 \in [0,1]
@jit
def getLogisticMapSequenceOriginal(x0, size):
    ret = np.zeros(size)
    ret[0] = x0
    for i in range(1, len(ret)):
        ret[i] = logisticMap(ret[i - 1])
    return ret


# x0 \in [0,1]
@jit
def getUniformLogisticMapSequenceOriginal(x0, size):
    ret = getLogisticMapSequenceOriginal(x0, size)
    return 2.0 * np.arcsin(np.sqrt(ret)) / np.pi


# x0 \in [-1,1]
@jit
def getSecondChebyshevPolynomialSequence(x0, size):
    ret = np.zeros(size)
    ret[0] = x0
    for i in range(1, len(ret)):
        ret[i] = 1 - 2 * ret[i - 1] ** 2

    # normalization
    print("np.mean(ret) = %f" % np.mean(ret))
    ret -= np.mean(ret)
    print("np.sqrt(np.var(ret)) = %f " % np.sqrt(np.var(ret)))
    ret /= np.sqrt(np.var(ret))

    return ret
