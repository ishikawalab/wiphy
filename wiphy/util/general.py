# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

__all__ = ['getGrayIndixes', 'frodiff', 'normsYHCodes', 'getEuclideanDistances', 'getMinimumEuclideanDistance', 'getDFTMatrix',
           'inv_dB', 'randn', 'randn_c', 'countErrorBits', 'getXORtoErrorBitsArray',
           'getErrorBitsTable', 'getRandomHermitianMatrix', 'convertIntToBinArray', 'CayleyTransform', 'CayleyTransformInv', 'asnumpy',
           'ascupy', 'frequencyToWavelength', 'kurtosis', 'testUnitaryCodes', 'isUnitary', 'toXpArray', 'dicToNumpy',
           'dicToDF', 'DFToDicNumpy', 'saveCSV', 'parseValue', 'argToDic']

import os
import re

import numpy as np
import pandas as pd
from numba import njit, int8
from scipy.constants import speed_of_light


@njit
def getGrayIndixes(bitWidth):
    return [i ^ (i >> 1) for i in range(2 ** bitWidth)]


@njit
def frodiff(x, y):
    return np.square(np.linalg.norm(x - y))


# Y in (N, T), H in (N, M), and codes in (Nc, M, T)
# numba doesn't support Y - H @ codes calculation
@njit# (parallel=True) worsens performance
def normsYHCodes(Y, H, codes):
    Nc = codes.shape[0]
    norms = np.zeros(Nc)
    for x in range(Nc):
        norms[x] = np.square(np.linalg.norm(Y - H @ codes[x]))

    return norms


# numba doesn't support broadcasting
@njit
def matmulb(H, codes):
    Nc = codes.shape[0]
    ret = np.zeros((Nc, H.shape[0], codes.shape[2])) + 0.j
    for x in range(Nc):
        ret[x] = H @ codes[x]

    return ret


@njit# (parallel=True) worsens performance
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


@njit# (parallel=True) worsens performance
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


@njit
def getDFTMatrix(N):
    W = np.zeros((N, N)) + 0.j
    omega = np.exp(2.0j * np.pi / N)
    for j in range(N):
        for k in range(N):
            W[j, k] = pow(omega, j * k)
    W /= np.sqrt(N)
    return W


@njit
def inv_dB(dB):
    return 10.0 ** (dB / 10.0)


@njit
def randn(*size):
    return np.random.normal(0, 1, size=size)


@njit
def randn_c(*size):
    """
    Generate an ndarray that follows the complex normal distribution.
    """
    return np.random.normal(0, 1 / np.sqrt(2.0), size=size) + np.random.normal(0, 1 / np.sqrt(2.0), size=size) * 1j


@njit
def countErrorBits(x, y):
    # return np.binary_repr(x ^ y).count('1') # not supported by numba
    ret = 0
    z = x ^ y

    while z >= 1:
        if z % 2 == 1:
            ret += 1
        z //= 2

    return ret


@njit
def getXORtoErrorBitsArray(Nc):
    # return xp.array(list(map(lambda x: bin(x).count('1'), range(Nc + 1))))
    ret = np.zeros(Nc + 1)
    for x in range(Nc + 1):
        ret[x] = countErrorBits(0, x)

    return ret


@njit
def getErrorBitsTable(Nc):
    errorTable = np.zeros((Nc, Nc), dtype=np.int8)
    for y in range(Nc):
        for x in range(y, Nc):
            errorTable[y][x] = errorTable[x][y] = countErrorBits(x, y)

    return errorTable


@njit
def getRandomHermitianMatrix(M):
    ret = np.diag(0j + randn(M))
    for y in range(0, M - 1):
        for x in range(y + 1, M):
            ret[y, x] = randn_c()
            ret[x, y] = np.conj(ret[y, x])
    return ret


@njit
def convertIntToBinArray(i, B):
    #return np.array(list(np.binary_repr(i).zfill(B))).astype(np.int) # does not compatible with numba
    ret = np.zeros(B, dtype=int8) # numba trick, numba.int8
    ret = ((i & (1 << np.arange(B)))) > 0
    return ret[::-1] # reverse


@njit
def CayleyTransform(H):
    M = H.shape[0]
    I = np.eye(M) + 0.j
    U = (I - 1.j * H) @ np.linalg.inv(I + 1.j * H)
    # U = np.matmul(H - 1.j * I, np.linalg.inv(H + 1.j * I))
    return U


@njit
def CayleyTransformInv(U):
    M = U.shape[0]
    I = np.eye(M) + 0.j
    # H = 1.j * np.matmul(I + U,np.linalg.inv(I - U))
    H = -1.j * np.linalg.inv(I + U) @ (I - U)
    return H


def asnumpy(xparr):
    if 'cupy' in str(type(xparr)):
        return np.asnumpy(xparr)  # cupy to numpy
    return xparr  # do nothing


def ascupy(nparr):
    if 'numpy' in str(type(nparr)):
        return np.asarray(nparr)  # numpy to cupy
    return nparr  # do nothing


# frequency [Hz], wavelength [m]
# I just wouldn't like to import speed_of_light in my script
@njit
def frequencyToWavelength(frequency):
    return speed_of_light / frequency


@njit
def kurtosis(x):
    mu = np.mean(x)
    mu2 = np.mean(np.power(x - mu, 2))
    mu4 = np.mean(np.power(x - mu, 4))
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


def testAlmostEqualBER(bera, berb):
    logbera = np.log10(asnumpy(bera))
    logberb = np.log10(asnumpy(berb))
    np.testing.assert_almost_equal(logbera, logberb, decimal=2)


# depricated
def toXpArray(arr):
    return np.asarray(arr)


def dicToNumpy(dic):
    for key in dic.keys():
        if 'cupy' in str(type(dic[key])):
            dic[key] = np.asnumpy(dic[key])
    return dic


def dicToDF(dic):
    for key in dic.keys():
        if 'cupy' in str(type(dic[key])):
            dic[key] = np.asnumpy(dic[key])
    return pd.DataFrame(dic)


def DFToDicNumpy(df):
    dic = {}
    for key in df.keys():
        dic[key] = np.array(df[key])
    return dic


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
