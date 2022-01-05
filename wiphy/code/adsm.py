# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

__all__ = ['generateADSMCodes', 'generateADSMDRCodes']

import numpy as np

from .modulator import generateAPSKSymbols


def _generateADSMCompanionMatrix(M, u_phase):
    A = np.zeros((M, M), dtype=np.complex)
    A[0, M - 1] = np.exp(1j * u_phase)
    for m in range(M - 1):
        A[(m + 1) % M, m] = 1

    return A


def generateADSMCodes(M, modtype, L):
    """
    Generates a codebook of the algebraic differential spatial modulation (ADSM), which was firstly proposed in [1] and was later extended in [2].

    - [1] R. Rajashekar, N. Ishikawa, S. Sugiura, K. V. S. Hari, and L. Hanzo, ``Full-diversity dispersion matrices from algebraic field extensions for differential spatial modulation,'' IEEE Trans. Veh. Technol., vol. 66, no. 1, pp. 385--394, 2017.
    - [2] R. Rajashekar, C. Xu, N. Ishikawa, S. Sugiura, K. V. S. Hari, and L. Hanzo, ``Algebraic differential spatial modulation is capable of approaching the performance of its coherent counterpart,'' IEEE Trans. Commun., vol. 65, no. 10, pp. 4260--4273, 2017.

    Args:
        M (int): the number of transmit antennas.
        modtype (string): the constellation type.
        L (int): the constellation size.
    """
    Nc = M * L
    symbols = generateAPSKSymbols(modtype, L)

    A = _generateADSMCompanionMatrix(M, 2.0 * np.pi / L)

    As = np.zeros((M, M, M), dtype=np.complex)  # M \times M \times M
    for m in range(M):
        As[m] = np.linalg.matrix_power(A, m)

    codestensor = np.kron(symbols, As)  # M \times M \times M * L (=Nc)
    return np.array(np.hsplit(np.hstack(codestensor), Nc))  # Nc \times M \times M


def generateADSMDRCodes(M, modtype, L, O):
    """
    Generates a codebook of the ADSM with multiple symbols.

    - [1] R. Rajashekar, N. Ishikawa, S. Sugiura, K. V. S. Hari, and L. Hanzo, ``Full-diversity dispersion matrices from algebraic field extensions for differential spatial modulation,'' IEEE Trans. Veh. Technol., vol. 66, no. 1, pp. 385--394, 2017.
    - [2] R. Rajashekar, C. Xu, N. Ishikawa, S. Sugiura, K. V. S. Hari, and L. Hanzo, ``Algebraic differential spatial modulation is capable of approaching the performance of its coherent counterpart,'' IEEE Trans. Commun., vol. 65, no. 10, pp. 4260--4273, 2017.

    Args:
        M (int): the number of transmit antennas.
        modtype (string): the constellation type.
        L (int): the constellation size.
        O (int): the number of symbols.
    """
    G = M // O  # diversity order
    Nc = O * (G * L) ** O
    Binner = int(np.log2(G * L))
    Bouter = int(np.log2(O))
    B = Bouter + O * Binner  # == np.log2(Nc)
    Cinner = generateADSMCodes(G, modtype, L)

    if O == 1:
        return Cinner

    N = _generateADSMCompanionMatrix(O, 2.0 * np.pi / L)
    Nd = np.kron(N, np.eye(G, dtype=np.complex))

    codes = np.zeros((Nc, M, M), dtype=np.complex)
    for i in range(Nc):
        bits = np.binary_repr(i, width=B)

        D = np.zeros((M, M), dtype=np.complex)
        for o in range(O):
            iin = int(bits[(o * Binner): (o * Binner + Binner)], 2)
            D[(o * G):(o * G + G), (o * G):(o * G + G)] = Cinner[iin]

        if G == 1:
            iou = 0
        else:
            iou = int(bits[(O * Binner):], 2)
        A = np.linalg.matrix_power(Nd, iou)

        codes[i] = D @ A

    return codes
