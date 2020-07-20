# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

__all__ = ['generateIMCodes']

import itertools

import numpy as np

from .modulator import generateAPSKSymbols
from ..util.im import convertIndsToMatrix, getIndexes


def generateIMCodes(indextype, M, K, Q, modtype, L, meanPower):
    """
    Generates a codebook of index modulation for M transmit antennas, where K antennas are activated based on the specified pattern of indextype. This method is applicable to the BLAST and the SIM cases. The number of codewords is Nc = Q * L^K.

    Args:
        indextype (string): dic, wen, mes, rand, and opt.
        M (int): the number of transmit antennas.
        K (int): the number of activated antennas.
        Q (int): the number of antenna activation patterns.
        modtype (string): the type of constellation, such as PSK, QAM, and SQAM.
        L (int): the constellation size.
        meanPower (float): the mean transmit power.

    Returns:
        ndarray: an (Nc, M, 1)-sized array of codewords.
    """

    apsksymbols = generateAPSKSymbols(modtype, L)
    kfoldsymbols = np.array(list(itertools.product(apsksymbols, repeat=K))).T
    inds = getIndexes(indextype, M, K, Q)
    indsm = convertIndsToMatrix(inds, M)
    codes = np.matmul(indsm, kfoldsymbols / np.sqrt(K))
    codes *= np.sqrt(meanPower)  # the mean power is normalized to meanPower
    codes = np.array(np.hsplit(np.hstack(codes), Q * kfoldsymbols.shape[1]))  # Nc \times M \times 1

    return codes
