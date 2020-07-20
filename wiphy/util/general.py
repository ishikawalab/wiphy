# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

__all__ = ['getGrayIndixes', 'frodiff', 'getEuclideanDistances', 'getMinimumEuclideanDistance', 'getDFTMatrix',
           'getDFTMatrixNumpy', 'inv_dB', 'randn', 'randn_c', 'countErrorBits', 'getXORtoErrorBitsArray',
           'getErrorBitsTable', 'getRandomHermitianMatrix', 'CayleyTransform', 'CayleyTransformInv', 'asnumpy',
           'ascupy', 'frequencyToWavelength', 'kurtosis', 'testUnitaryCodes', 'isUnitary', 'toXpArray', 'dicToNumpy',
           'dicToDF', 'saveCSV', 'parseValue', 'argToDic']

import os
import re

import numpy as np
import pandas as pd
from numba import jit
from scipy.constants import speed_of_light
from sympy.combinatorics.graycode import GrayCode

if os.getenv("USECUPY") == "1":
    import cupy as xp
else:
    import numpy as xp


def getGrayIndixes(bitWidth):
    gray = GrayCode(bitWidth)
    return [int(strb, 2) for strb in gray.generate_gray()]


def frodiff(x, y):
    return xp.square(xp.linalg.norm(x - y))


@jit(parallel=True)
def getEuclideanDistances(codes):
    # The following straightforward implementation with numba is the fastest
    Nc, M, T = codes.shape[0], codes.shape[1], codes.shape[2]
    tolBase = 2.22e-16 * max(M, T)

    ret = np.zeros(int(Nc * (Nc - 1) / 2))
    i = 0
    for y in range(0, Nc):
        for x in range(y + 1, Nc):
            diff = codes[y] - codes[x]
            _, s, _ = np.linalg.svd(diff.dot(np.conj(diff.T)))
            ret[i] = np.prod(s[s > tolBase])
            i += 1
    return ret


@jit(parallel=True)
def getMinimumEuclideanDistance(codes):
    # The following straightforward implementation with numba is the fastest
    Nc, M, T = codes.shape[0], codes.shape[1], codes.shape[2]
    tolBase = 2.22e-16 * max(M, T)
    mind = np.inf
    for y in range(0, Nc):
        for x in range(y + 1, Nc):
            diff = codes[y] - codes[x]
            _, s, _ = np.linalg.svd(diff.dot(np.conj(diff.T)))
            d = np.prod(s[s > tolBase])
            if d < mind:
                mind = d
    return mind


def getDFTMatrix(N):
    W = xp.zeros((N, N), dtype=complex)
    omega = xp.exp(2.0j * xp.pi / N)
    for j in range(N):
        for k in range(N):
            W[j, k] = pow(omega, j * k)
    W /= xp.sqrt(N)
    return W


def getDFTMatrixNumpy(N):
    W = np.zeros((N, N), dtype=complex)
    omega = np.exp(2.0j * np.pi / N)
    for j in range(N):
        for k in range(N):
            W[j, k] = pow(omega, j * k)
    W /= np.sqrt(N)
    return W


def inv_dB(dB):
    return 10.0 ** (dB / 10.0)


def randn(*size):
    return xp.random.normal(0, 1, size=size)


def randn_c(*size):
    """
    Generate an ndarray that follows the complex normal distribution.
    """
    return xp.random.normal(0, 1 / xp.sqrt(2.0), size=size) + xp.random.normal(0, 1 / xp.sqrt(2.0), size=size) * 1j


def countErrorBits(x, y):
    return bin(x ^ y).count('1')


def getXORtoErrorBitsArray(Nc):
    # return xp.array(list(map(lambda x: bin(x).count('1'), range(Nc + 1))))
    ret = xp.zeros(Nc + 1)
    for x in range(Nc + 1):
        ret[x] = bin(x).count('1')

    return ret


def getErrorBitsTable(Nc):
    errorArray = getXORtoErrorBitsArray(Nc)
    errorTable = xp.zeros((Nc, Nc), dtype=xp.int8)
    for y in range(Nc):
        for x in range(y, Nc):
            errorTable[y][x] = errorTable[x][y] = errorArray[x ^ y]

    return errorTable


def getRandomHermitianMatrix(M):
    ret = xp.diag(0j + randn(M))
    for y in range(0, M - 1):
        for x in range(y + 1, M):
            ret[y, x] = randn_c()
            ret[x, y] = xp.conj(ret[y, x])
    return ret


# TODO: need to support cp
def CayleyTransform(H):
    M = H.shape[0]
    I = np.eye(M, dtype=np.complex)
    U = np.matmul(I - 1.j * H, np.linalg.inv(I + 1.j * H))
    # U = np.matmul(H - 1.j * I, np.linalg.inv(H + 1.j * I))
    return U


# TODO: need to support cp
def CayleyTransformInv(U):
    M = U.shape[0]
    I = np.eye(M, dtype=np.complex)
    # H = 1.j * np.matmul(I + U,np.linalg.inv(I - U))
    H = -1.j * np.matmul(np.linalg.inv(I + U), I - U)
    return H


def asnumpy(xparr):
    if 'cupy' in str(type(xparr)):
        return xp.asnumpy(xparr)  # cupy to numpy
    return xparr  # do nothing


def ascupy(nparr):
    if 'numpy' in str(type(nparr)):
        return xp.asarray(nparr)  # numpy to cupy
    return nparr  # do nothing


# frequency [Hz], wavelength [m]
@jit
def frequencyToWavelength(frequency):
    return speed_of_light / frequency


def kurtosis(x):
    mu = xp.mean(x)
    mu2 = xp.mean(xp.power(x - mu, 2))
    mu4 = xp.mean(xp.power(x - mu, 4))
    beta2 = mu4 / (mu2 * mu2) - 3.0
    return beta2


def testUnitaryCodes(codes):
    Nc, M, _ = codes.shape
    codes = codes.reshape(-1, M)
    np.testing.assert_almost_equal(codes.T.conj() @ codes / Nc, np.eye(M))


def isUnitary(codes):
    if len(codes.shape) == 2:
        M, _ = codes.shape
        codes = codes.reshape(-1, M)
        return frodiff(codes.T.conj() @ codes, np.eye(M)) < 1e-6
    elif len(codes.shape) == 3:
        Nc, M, _ = codes.shape
        codes = codes.reshape(-1, M)
        return frodiff(codes.T.conj() @ codes / Nc, np.eye(M)) < 1e-6


def toXpArray(arr):
    return xp.asarray(arr)


def dicToNumpy(dic):
    for key in dic.keys():
        if 'cupy' in str(type(dic[key])):
            dic[key] = xp.asnumpy(dic[key])
    return dic


def dicToDF(dic):
    for key in dic.keys():
        if 'cupy' in str(type(dic[key])):
            dic[key] = xp.asnumpy(dic[key])
    return pd.DataFrame(dic)


def saveCSV(filename, df):
    if not os.path.exists("results/"):
        os.mkdir("results/")
    fname = "results/" + filename + ".csv"
    if 'dic' in str(type(df)):
        df = pd.DataFrame(df)
    df.to_csv(fname, index=False, float_format="%.20e")
    # np.savetxt(fname, np.c_[x, y], delimiter = ",", header = xlabel + "," + ylabel)
    print("The result was saved to " + fname)


def parseValue(value):
    """Converts a string into a float, int, or string parameter."""
    if re.match(r'-*\d+', value):
        if value.find("e") > 0:
            return float(value)  # e.g. IT=1e5, IT=2.5e7
        elif value.find(".") > 0:
            return float(value)  # e.g. to=50.00
        else:
            return int(value)  # e.g. M=4
    else:
        return value  # e.g. channel=rayleigh


def argToDic(arg):
    """
    Converts a parameter sequence into a dict.

    Args:
        arg (string): specified simulation parameters."""
    params = dict()

    options = arg.split("_")
    if "=" in options[0]:
        params["mode"] = ""
    else:
        params["mode"] = options.pop(0)

    # parse arguments such as "M=2"
    for op in options:
        pair = op.split("=")
        pv = parseValue(pair[1])

        # exception
        if "IT" in pair[0]:
            pv = int(pv)

        params[pair[0]] = pv

    return params
