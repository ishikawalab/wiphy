# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

__all__ = ['generateOSTBCodes']

import itertools

import numpy as np

from .modulator import generateAPSKSymbols


def generateOSTBCodes(M, modtype, L, nsymbols=1):
    """
    Generates a codebook of orthogonal space-time block code (OSTBC). A seminal research can be found in [1].

    - [1] S. Alamouti, ``A simple transmit diversity technique for wireless communications,'' IEEE J. Sel. Areas Commun., vol. 16, no. 8, pp. 1451--1458, 1998.
    
    Args:
        M (int): the number of transmit antennas.
        modtype (string): the constellation type, e.g. PSK, QAM, SQAM, and PAM.
        L (int): the constellation size.
        nsymbols (int): the number of embedded symbols.
    """
    if M == 2:
        nsymbols = M

    elif M == 4:
        if modtype == "PAM":
            nsymbols = M
        else:
            if nsymbols == 1:
                print("Please specify nsymbols = 2, 3 or 4. I use nsymbols = 2.")
                nsymbols = 2
            if modtype == "QAM" or nsymbols == 4:
                print("Note that the space-time codewords become non-orthogonal.")
    elif M == 8:
        if modtype == "PAM":
            nsymbols = M
        else:
            print("OSTBC with M=8 and PSK is not supported")
    elif M == 16:
        nsymbols = M

    B = nsymbols * np.log2(L)
    Nc = int(2 ** B)

    # initialize codes
    symbols = generateAPSKSymbols(modtype, L)
    kfoldsymbols = np.array(list(itertools.product(symbols, repeat=nsymbols)))  # L^nsymbols \times nsymbols

    codes = np.zeros((Nc, M, M), dtype=complex)
    for i in range(kfoldsymbols.shape[0]):
        s = kfoldsymbols[i, :]

        if M == 2:
            codes[i] = [[s[0], s[1]], [-np.conj(s[1]), np.conj(s[0])]]
            codes[i] /= np.sqrt(nsymbols)

        if M == 4:
            if modtype == "PAM":
                codes[i, 0, 0] = s[0]
                codes[i, 0, 1] = s[1]
                codes[i, 0, 2] = s[2]
                codes[i, 0, 3] = s[3]
                codes[i, 1, 0] = -s[1]
                codes[i, 1, 1] = s[0]
                codes[i, 1, 2] = -s[3]
                codes[i, 1, 3] = s[2]
                codes[i, 2, 0] = -s[2]
                codes[i, 2, 1] = s[3]
                codes[i, 2, 2] = s[0]
                codes[i, 2, 3] = -s[1]
                codes[i, 3, 0] = -s[3]
                codes[i, 3, 1] = -s[2]
                codes[i, 3, 2] = s[1]
                codes[i, 3, 3] = s[0]
                codes[i] /= np.sqrt(nsymbols)
            else:
                if nsymbols == 2:
                    codes[i, 0, 0] = s[0]
                    codes[i, 0, 1] = s[1]
                    codes[i, 1, 0] = -np.conj(s[1])
                    codes[i, 1, 1] = np.conj(s[0])
                    codes[i, 2, 2] = s[0]
                    codes[i, 2, 3] = s[1]
                    codes[i, 3, 2] = -np.conj(s[1])
                    codes[i, 3, 3] = np.conj(s[0])
                    codes[i] /= np.sqrt(nsymbols)
                elif nsymbols == 3:
                    codes[i, 0, 0] = s[0]
                    codes[i, 0, 1] = s[1]
                    codes[i, 0, 2] = s[2]
                    codes[i, 1, 0] = -np.conj(s[1])
                    codes[i, 1, 1] = np.conj(s[0])
                    codes[i, 1, 3] = s[2]
                    codes[i, 2, 0] = np.conj(s[2])
                    codes[i, 2, 2] = -np.conj(s[0])
                    codes[i, 2, 3] = s[1]
                    codes[i, 3, 1] = np.conj(s[2])
                    codes[i, 3, 2] = -np.conj(s[1])
                    codes[i, 3, 3] = -s[0]
                    codes[i] /= np.sqrt(nsymbols)
                elif nsymbols == 4:
                    codes[i, 0, 0] = s[0]
                    codes[i, 0, 1] = -np.conj(s[1])
                    codes[i, 1, 0] = s[1]
                    codes[i, 1, 1] = np.conj(s[0])
                    codes[i, 0, 2] = s[2]
                    codes[i, 0, 3] = -np.conj(s[3])
                    codes[i, 1, 2] = s[3]
                    codes[i, 1, 3] = np.conj(s[2])
                    codes[i, 2, 0] = s[2]
                    codes[i, 2, 1] = -np.conj(s[3])
                    codes[i, 3, 0] = s[3]
                    codes[i, 3, 1] = np.conj(s[2])
                    codes[i, 2, 2] = s[0]
                    codes[i, 2, 3] = -np.conj(s[1])
                    codes[i, 3, 2] = s[1]
                    codes[i, 3, 3] = np.conj(s[0])
                    codes[i] /= np.sqrt(nsymbols)

        elif M == 8:
            if modtype == "PAM":
                codes[i, 0] = [+s[0], +s[1], +s[2], +s[3], +s[4], +s[5], +s[6], +s[7]]
                codes[i, 1] = [-s[1], +s[0], +s[3], -s[2], +s[5], -s[4], -s[7], +s[6]]
                codes[i, 2] = [-s[2], -s[3], +s[0], +s[1], +s[6], +s[7], -s[4], -s[5]]
                codes[i, 3] = [-s[3], +s[2], -s[1], +s[0], +s[7], -s[6], +s[5], -s[4]]
                codes[i, 4] = [-s[4], -s[5], -s[6], -s[7], +s[0], +s[1], +s[2], +s[3]]
                codes[i, 5] = [-s[5], +s[4], -s[7], +s[6], -s[1], +s[0], -s[3], +s[2]]
                codes[i, 6] = [-s[6], +s[7], +s[4], -s[5], -s[2], +s[3], +s[0], -s[1]]
                codes[i, 7] = [-s[7], -s[6], +s[5], +s[4], -s[3], -s[2], +s[1], +s[0]]
                codes[i] /= np.sqrt(nsymbols)

        elif M == 16:
            for k in range(8):
                codes[i, 0 + k * 2, 0 + k * 2] = s[0 + k * 2]
                codes[i, 0 + k * 2, 1 + k * 2] = s[1 + k * 2]
                codes[i, 1 + k * 2, 0 + k * 2] = -np.conj(s[1 + k * 2])
                codes[i, 1 + k * 2, 1 + k * 2] = np.conj(s[0 + k * 2])
            codes[i] /= np.sqrt(2)

    return codes
