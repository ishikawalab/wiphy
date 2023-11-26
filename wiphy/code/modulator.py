# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

__all__ = ['generatePSKSymbols', 'generateQAMSymbols', 'generateStarQAMSymbols', 'generateAPSKSymbols',
           'generateSymbolCodes']

import numpy as np

from ..util.general import getGrayIndixes


def generatePSKSymbols(constellationSize=2):
    """Generates phase shift keying symbols having L = constellationSize. Returns an L-sized array."""
    L = constellationSize
    bitWidth = np.log2(L)

    if bitWidth != np.floor(bitWidth):
        print("The specified constellationSize is not a power of two")
        return np.array([], dtype=complex)

    if L == 1:
        return np.array([1.], dtype=complex)

    grayIndexes = getGrayIndixes(bitWidth)
    originalSymbols = np.exp(2.0j * np.pi * np.arange(L) / L)
    # We would like to avoid quantization errors
    l4 = np.min([L, 4])
    indsAxis = (np.arange(l4) * L / l4).astype(int)
    originalSymbols[indsAxis] = np.rint(originalSymbols[indsAxis])

    retSymbols = np.zeros(len(originalSymbols), dtype=complex)
    for i, g in enumerate(grayIndexes):
        retSymbols[g] = originalSymbols[i]

    return retSymbols


def generateQAMSymbols(constellationSize=4):
    """Generates quadrature amplitude modulation symbols having L = constellationSize. Returns an L-sized array."""
    L = constellationSize
    sqrtL = np.floor(np.sqrt(L))

    if sqrtL * sqrtL != L:
        print("The specified constellationSize is not an even power of two")
        return np.array([], dtype=complex)

    sigma = np.sqrt((L - 1) * 2 / 3)
    y = np.floor(np.arange(L) / sqrtL)
    x = np.arange(L) % sqrtL
    originalSymbols = ((sqrtL - 1) - 2 * x) / sigma + 1.j * ((sqrtL - 1) - 2 * y) / sigma

    logsqL = np.floor(np.log2(sqrtL))
    grayIndexes = getGrayIndixes(logsqL)
    grayIndexes = (np.take(grayIndexes, list(y)) * 2 ** logsqL + np.take(grayIndexes, list(x))).astype(int)

    return np.take(originalSymbols, grayIndexes)


def generateStarQAMSymbols(constellationSize=2):
    """Generates star quadrature amplitude modulation symbols having L = constellationSize. Returns an L-sized array.

    - [1] W. T. Webb, L. Hanzo, and R. Steele, "Bandwidth efficient QAM schemes for Rayleigh fading channels," IEE Proc., vol. 138, no. 3, pp. 169--175, 1991.
    """
    L = constellationSize
    p = np.log2(L) / 2 - 1
    subConstellationSize = int(4 * 2 ** np.floor(p))
    Nlevels = int(2 ** np.ceil(p))

    sigma = np.sqrt(6.0 / (Nlevels + 1.0) / (2.0 * Nlevels + 1.0))
    symbols = np.zeros(L, dtype=complex)
    for level_id in range(Nlevels):
        subpsk = generatePSKSymbols(subConstellationSize)
        # symbols.append((1.0 + level_id) * sigma * mod.symbols)
        symbols[(level_id * subConstellationSize):((level_id + 1) * subConstellationSize)] = (
                                                                                                     1.0 + level_id) * sigma * subpsk
    return symbols


def generateAPSKSymbols(mode="PSK", constellationSize=2):
    """Generates a general constellation such as PSK, QAM, and star-QAM (SQAM). Returns an L-sized array.

    Args:
        mode (string): the type of constellation, such as PSK, QAM, and SQAM.
        constellationSize (int): the constellation size.

    Returns:
        ndarray: an array of symbols whose length is constellationSize.
    """

    if mode == "PSK":
        return generatePSKSymbols(constellationSize)
    elif mode == "QAM":
        return generateQAMSymbols(constellationSize)
    elif mode == "SQAM" or mode == "StarQAM":
        return generateStarQAMSymbols(constellationSize)


def generateSymbolCodes(mode="PSK", constellationSize=2):
    return generateAPSKSymbols(mode, constellationSize).reshape(constellationSize, 1, 1)
