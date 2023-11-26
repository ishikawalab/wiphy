# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

__all__ = ['generateTASTCodes', 'searchTASTFactors']

import numpy as np

from .modulator import generatePSKSymbols, generateStarQAMSymbols


def generateTASTCodes(M, Q, L, modtype="SQAM"):
    """
    Differential space-time shift keying using threaded algebraic space-time (DSTSK-TAST) coding, which was proposed in [1]. Other relevant papers are found in [2,3].

    - [1] C. Xu, P. Zhang, R. Rajashekar, N. Ishikawa, S. Sugiura, L. Wang, and L. Hanzo, ``Finite-cardinality single-RF differential space-time modulation for improving the diversity-throughput tradeoff,'' IEEE Trans. Commun. Press, vol. 67, no. 1, pp. 318--335, 2019.
    - [2] C. Xu, R. Rajashekar, N. Ishikawa, S. Sugiura, and L. Hanzo, ``Single-RF index shift keying aided differential space-time block coding,'' IEEE Trans. Signal Process., vol. 66, no. 3, pp. 773--788, 2018.
    - [3] C. Xu, P. Zhang, R. Rajashekar, N. Ishikawa, S. Sugiura, Z. Wang, and L. Hanzo, ````Near-perfect'' finite-cardinality generalized space-time shift keying,'' IEEE J. Sel. Areas Commun., in press.

    Args:
        M (int): the number of transmit antennas.
        Q (int): the number of dispersion elements.
        L (int): the constellation size.
        modtype (str): the constellation type.
    """
    if modtype == "SQAM":
        apsksymbols = generateStarQAMSymbols(L)
        p = np.log2(L) / 2 - 1
        La = int(2 ** np.ceil(p))
        Lp = int(4 * 2 ** np.floor(p))
    elif modtype == "PSK":
        apsksymbols = generatePSKSymbols(L)
        La = 1
        Lp = L

    G = np.zeros((M, M), dtype=complex)
    G[0, M - 1] = 1
    for m in range(M - 1):
        G[m + 1, m] = 1

    msymbols = np.exp(1j * 2.0 * np.pi * np.arange(M) / (Lp * np.max([M, Q])))
    u = _getDiversityMaximizingFactors(M, Q, La, Lp)

    codes = np.zeros((M * L * Q, M, M), dtype=complex)
    for l in range(L):
        for m in range(M):
            for q in range(Q):
                qsymbols = np.exp(1j * 2.0 * np.pi * q * u * np.arange(M) / (L * Q))
                codes[Q * M * l + m * Q + q] = np.matmul(apsksymbols[l] * msymbols[m] * np.linalg.matrix_power(G, m),
                                                         np.diag(qsymbols))

    return codes


def searchTASTFactors(M, Q, Lp):
    maxp = -1

    while True:
        u = np.random.randint(Q * Lp, size=M)
        # print(u)
        p = _evaluate(M, Q, Lp, u)
        if p > maxp:
            maxp = p
            bestu = u
            print("if M == %d and Q == %d and La == 1 and Lp == %d:" % (M, Q, Lp))
            print("    # rate = %f, maxp = %e" % (np.log2(M * Q * Lp) / M, maxp))
            print("    u = " + np.array2string(u, separator=',', max_line_width=np.inf))


def _evaluate(M, Q, Lp, u):
    LDM = Lp * Q
    pmin = np.inf
    for l in range(2, Lp + 1):
        for q in range(2, Q + 1):
            p = np.prod(np.power(np.abs(np.sin(np.pi * (q * u - u + l * Q) / LDM)), 1 / M))
            if p < pmin:
                pmin = p

    return pmin


def _getDiversityMaximizingFactors(M, Q, La, Lp):
    if Q == 1:
        # qsymbols = [0, ..., 0] in this case
        return np.ones(M)

    if M == 2 and Q == 1 and La == 1 and Lp == 2:
        # Rate = 1
        u = [1, 1]
    elif M == 2 and Q == 2 and La == 1 and Lp == 2:
        # rate = 1.500000, maxp = 1.000000e+00
        u = [2, 2]
    elif M == 2 and Q == 4 and La == 1 and Lp == 2:
        # rate = 2.000000, maxp = 7.071068e-01
        u = [6, 6]
    elif M == 2 and Q == 2 and La == 1 and Lp == 8:
        # rate = 2.500000, maxp = 4.374262e-01
        u = [7, 15]
    elif M == 2 and Q == 2 and La == 2 and Lp == 8:
        # Rate = 3
        # u = [3,13] # [1, p. 22]
        u = [2, 14]
    elif M == 2 and Q == 4 and La == 1 and Lp == 8:
        # Rate = 3
        u = [1, 7]
    elif M == 2 and Q == 4 and La == 2 and Lp == 8:
        # Rate = 3.5
        u = [29, 3]
    elif M == 2 and Q == 4 and La == 1 and Lp == 16:
        # rate = 3.500000, maxp = 1.950903e-01
        u = [30, 62]
    elif M == 2 and Q == 4 and La == 4 and Lp == 8:  # Rate = 4, maxp = 0.080042
        u = [29, 3]
    elif M == 2 and Q == 8 and La == 2 and Lp == 8:  # Rate = 4, maxp = 0.0957012
        u = [53, 27]
    elif M == 2 and Q == 16 and La == 1 and Lp == 8:
        # Rate = 4, maxp = 0.109983
        # u = [51, 81]
        # rate = 4.000000, maxp = 1.888190e-01
        u = [94, 46]
    elif M == 2 and Q == 8 and La == 1 and Lp == 16:  # Rate = 4, maxp = 0.140074
        u = [3, 117]
    elif M == 2 and Q == 4 and La == 4 and Lp == 16:  # Rate = 4.5
        u = [7, 57]
    elif M == 2 and Q == 8 and La == 4 and Lp == 16:  # rate = 5, maxp = 0.0511476
        u = [108, 52]
    elif M == 2 and Q == 16 and La == 4 and Lp == 8:  # rate = 5, maxp = 0.04216
        u = [39, 12]
    elif M == 2 and Q == 32 and La == 2 and Lp == 8:  # rate = 5, maxp = 0.0483617
        u = [242, 46]
    elif M == 2 and Q == 8 and La == 8 and Lp == 16:  # Rate = 5.5
        u = [3, 117]
    elif M == 2 and Q == 8 and La == 8 and Lp == 32:  # Rate = 6
        u = [11, 237]
    elif M == 4 and Q == 1 and La == 1 and Lp == 2:  # Rate = 0.75
        u = [1, 1, 1, 1]
    elif M == 4 and Q == 2 and La == 1 and Lp == 2:  # Rate = 1
        u = [1, 1, 3, 3]
    elif M == 4 and Q == 2 and La == 1 and Lp == 4:  # Rate = 1.25
        u = [1, 3, 7, 5]
    elif M == 4 and Q == 2 and La == 1 and Lp == 8:  # Rate = 1.5
        # u = [1,3,7,9]
        u = [1, 5, 11, 15]
    elif M == 4 and Q == 4 and La == 1 and Lp == 8:  # Rate = 1.75
        # u = [9,21,15,27]
        u = [5, 25, 11, 31]
    elif M == 4 and Q == 1 and La == 4 and Lp == 16:  # rate = 2, maxp = 0.0712369
        u = [0, 0, 0, 0]
    elif M == 4 and Q == 1 and La == 1 and Lp == 64:  # rate = 2, maxp = 0.0490677
        u = [0, 0, 0, 0]
    elif M == 4 and Q == 2 and La == 4 and Lp == 8:  # rate = 2, maxp = 0.0969766
        u = [2, 13, 3, 14]
    elif M == 4 and Q == 2 and La == 1 and Lp == 32:
        # rate = 2, maxp = 0.0980171
        # u = [59, 31, 49, 41]
        # rate = 2.000000, maxp = 3.951722e-01
        u = [35, 3, 19, 51]
    elif M == 4 and Q == 4 and La == 1 and Lp == 16:
        # rate = 2, maxp = 0.19509
        # u = [37, 27, 11, 21]
        # rate = 2.000000, maxp = 3.732889e-01
        u = [13, 29, 41, 63]
    elif M == 4 and Q == 4 and La == 2 and Lp == 8:  # rate = 2, maxp = 0.214639
        u = [21, 3, 29, 11]
    elif M == 4 and Q == 8 and La == 1 and Lp == 8:
        # rate = 2, maxp = 0.312322
        # u = [15, 41, 57, 63]
        # rate = 2.000000, maxp = 3.826834e-01
        u = [14, 30, 46, 62]
    elif M == 4 and Q == 8 and La == 2 and Lp == 4:  # rate = 2, maxp = 0.199766
        u = [26, 9, 31, 14]
    elif M == 4 and Q == 16 and La == 1 and Lp == 4:
        # rate = 2, maxp = 0.256578
        u = [31, 61, 49, 59]
        # rate = 2.000000, maxp = 3.660978e-01
        u = [62, 54, 46, 14]
    elif M == 4 and Q == 32 and La == 1 and Lp == 2:
        # rate = 2, maxp = 0.312322
        u = [51, 21, 5, 3]
        # rate = 2.000000, maxp = 3.826834e-01
        u = [2, 30, 50, 46]
    elif M == 4 and Q == 16 and La == 1 and Lp == 8:  # Rate = 2.25
        u = [21, 37, 91, 83]
    elif M == 4 and Q == 8 and La == 8 and Lp == 16:  # rate = 3, maxp = 0.0260332
        u = [114, 25, 14, 99]
    elif M == 4 and Q == 16 and La == 4 and Lp == 16:  # rate = 3, maxp = 0.0546157
        u = [97, 145, 236, 25]
    elif M == 4 and Q == 32 and La == 4 and Lp == 8:  # rate = 3, maxp = 0.0557874
        u = [196, 108, 252, 65]
    elif M == 4 and Q == 64 and La == 1 and Lp == 16:  # Rate = 3
        u = [633, 603, 559, 797]
    elif M == 8 and Q == 1 and La == 2 and Lp == 8:  # rate = 0.875, maxp = 0.24203
        u = [0, 0, 0, 0, 0, 0, 0, 0]
    elif M == 8 and Q == 2 and La == 2 and Lp == 4:  # rate = 0.875, maxp = 0.298721
        u = [1, 1, 7, 2, 7, 6, 1, 7]
    elif M == 8 and Q == 4 and La == 1 and Lp == 4:  # rate = 0.875, maxp = 0.522137
        u = [9, 11, 3, 7, 15, 13, 5, 1]
    elif M == 8 and Q == 2 and La == 2 and Lp == 8:  # rate = 1, maxp = 0.24203
        u = [9, 1, 1, 15, 14, 1, 15, 10]
    elif M == 8 and Q == 4 and La == 2 and Lp == 4:  # rate = 1, maxp = 0.262553
        u = [1, 11, 2, 11, 14, 5, 5, 15]
    elif M == 8 and Q == 8 and La == 1 and Lp == 4:  # rate = 1, maxp = 0.375254
        u = [25, 31, 15, 13, 7, 1, 17, 3]
    elif M == 16 and Q == 1 and La == 1 and Lp == 16:  # rate = 0.5, maxp = 0.19509
        u = [7, 15, 0, 3, 8, 10, 1, 11, 15, 15, 14, 2, 5, 4, 14, 8]
    elif M == 16 and Q == 2 and La == 1 and Lp == 8:
        # rate = 0.500000, maxp = 5.204288e-01
        u = [1, 14, 7, 14, 3, 5, 3, 9, 13, 7, 13, 7, 13, 13, 15, 3]
    elif M == 16 and Q == 4 and La == 1 and Lp == 4:
        # rate = 0.5, maxp = 0
        # u = [4,10,8,10,14,8,2,0,6,9,11,11,6,4,0,1]
        # rate = 0.500000, maxp = 5.571934e-01
        u = [3, 5, 13, 14, 6, 7, 11, 15, 6, 1, 9, 15, 9, 5, 14, 13]
    elif M == 16 and Q == 8 and La == 1 and Lp == 2:
        # rate = 0.500000, maxp = 5.693943e-01
        u = [6, 2, 11, 10, 11, 15, 2, 3, 3, 15, 7, 2, 10, 14, 9, 6]
    elif M == 16 and Q == 64 and La == 1 and Lp == 1:  # rate = 0.625, maxp = 0
        u = [63, 0, 18, 34, 62, 36, 37, 1, 26, 55, 37, 54, 46, 15, 39, 26]
    elif M == 64 and Q == 2 and La == 1 and Lp == 2:  # rate = 0.125, maxp = 0
        u = [0, 1, 3, 0, 2, 2, 1, 3, 1, 2, 0, 0, 0, 1, 0, 3, 2, 2, 0, 2, 1, 2, 1, 1, 0, 1, 2, 2, 0, 2, 0, 1, 0, 3, 1, 2,
             3, 3, 3, 2, 3, 1, 2, 1, 1, 0, 3, 3, 2, 0, 0, 0, 3, 3, 3, 2, 2, 3, 0, 1, 2, 0, 1, 0]
    elif M == 64 and Q == 4 and La == 1 and Lp == 1:  # rate = 0.125, maxp = 0
        u = [0, 3, 2, 2, 1, 0, 1, 2, 0, 3, 1, 0, 0, 3, 0, 1, 2, 2, 0, 0, 3, 2, 1, 2, 2, 3, 0, 2, 3, 1, 3, 2, 2, 2, 3, 2,
             3, 0, 0, 3, 0, 0, 3, 1, 0, 3, 3, 3, 2, 2, 3, 2, 2, 1, 3, 2, 0, 2, 3, 1, 3, 1, 1, 2]
    elif M == 64 and Q == 1 and La == 1 and Lp == 4:  # rate = 0.125, maxp = 0.502715
        u = [2, 1, 1, 2, 1, 2, 0, 0, 1, 2, 1, 1, 2, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 2, 3, 2, 1, 3, 3, 1, 1, 1,
             3, 3, 2, 0, 1, 2, 2, 2, 1, 2, 3, 2, 3, 0, 2, 0, 2, 1, 2, 0, 2, 0, 0, 1, 0, 2, 2, 1]
    else:
        print("TASTCode.py does not support the given parameters M = %d, Q = %d, La = %d, and Lp = %d" % (
            M, Q, La, Lp))
        u = np.zeros(M)
    return np.array(u)
