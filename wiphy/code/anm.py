# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

__all__ = ['generateANMCodes']

import numpy as np

from .modulator import generateAPSKSymbols


def generateANMCodes(M, modtype, L):
    symbols = generateAPSKSymbols(modtype, L)
    codes = np.zeros((M * L, M, 1), dtype=np.complex)

    for m in range(M):
        for l in range(L):
            for k in range(m + 1):
                codes[m * L + l, k, 0] = symbols[l] / np.sqrt(m + 1)

    return codes
