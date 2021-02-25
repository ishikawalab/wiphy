# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import sys
from numpy import *
from tqdm import trange
from wiphy.channel.ideal import generateRayleighChannel
from wiphy.code import generateCodes
from wiphy.util.general import getXORtoErrorBitsArray, inv_dB, randn_c, argToDic, saveCSV


def simulateBERReference(codes, channelfun, params, printValue=True):
    """
    Simulates BER values at multiple SNRs, where the straightforward reference algorithm is used. Note that this time complexity is unrealistically high. This simulation relies on the non-coherent maximum likelihood detector, where semi-unitary matrices are used for differential encoding and decdoing. The semi-unitary matrix is defined by $U U^H = \alpha I_M$ and $\alpha \in \mathbb{R}$. The environment variable USECUPY determines whether to use cupy or not.

    Args:
        codes (ndarray): an input codebook, which is generated on the CPU memory and is transferred into the GPU memory.
        channelfun (function): .
        params (dict): simulation parameters.
        printValue (bool): a flag that determines whether to print the simulated values.

    Returns:
        ret (dict): a dict that has two keys: snr_dB and ber, and contains the corresponding results. All the results are transferred into the CPU memory.
    """

    IT, M, N, Nc, B = params["IT"], params["M"], params["N"], codes.shape[0], log2(codes.shape[0])
    snr_dBs = linspace(params["from"], params["to"], params["len"])
    sigmav2s = 1.0 / inv_dB(snr_dBs)
    xor2ebits = getXORtoErrorBitsArray(Nc)

    bers = zeros(len(snr_dBs))
    for i in trange(len(snr_dBs)):
        errorBits = 0
        v0 = randn_c(N, M) * sqrt(sigmav2s[i])  # N \times M
        s0 = eye(M, dtype=complex)
        rs0 = s0  # the estimated codeword at the receiver
        currentBeta = linalg.norm(rs0)  # the estimated normalizing factor at the receiver
        for _ in range(IT):
            codei = random.randint(0, Nc)
            s1 = matmul(s0, codes[codei]) / linalg.norm(s0)  # semi-unitary differential encoding

            h = channelfun(N, M)  # N \times M
            v1 = randn_c(N, M) * sqrt(sigmav2s[i])  # N \times M

            y0 = matmul(h, s0) + v0  # N \times M
            y1 = matmul(h, s1) + v1  # N \times M

            # semi-unitary non-coherent detection
            p = square(abs(y1 - matmul(y0, codes) / currentBeta))  # Nc \times N \times M
            norms = sum(p, axis=(1, 2))  # summation over the (N,M) axes
            mini = argmin(norms)
            errorBits += sum(xor2ebits[codei ^ mini])

            rs1 = matmul(rs0, codes[mini]) / currentBeta
            currentBeta = linalg.norm(rs1)
            v0 = v1
            s0 = s1
            rs0 = rs1

        bers[i] = errorBits / (IT * B)
        if printValue:
            print("At SNR = %1.2f dB, BER = %d / %d = %1.10e" % (snr_dBs[i], errorBits, IT * B, bers[i]))

    return {"snr_dB": snr_dBs, "ber": bers}


if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = sys.argv[1:]
    else:
        args = [
            "BER_sim=sudiff_channel=rayleigh_code=symbol_M=1_N=2_L=16_mod=SQAM_IT=1e4_from=0.00_to=40.00_len=21",
            "BER_sim=sudiff_channel=rayleigh_code=OSTBC_M=2_N=2_T=2_L=16_mod=SQAM_IT=1e4_from=0.00_to=40.00_len=21"
        ]

    for arg in args:
        print("Simulating arg = " + arg)
        params = argToDic(arg)
        codes = generateCodes(params)
        ret = simulateBERReference(codes, generateRayleighChannel, params, printValue=False)
        saveCSV(arg, ret)
        print(ret)
