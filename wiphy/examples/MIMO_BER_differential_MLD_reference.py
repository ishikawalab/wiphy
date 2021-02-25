# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import sys
from numpy import *
from tqdm import trange
from wiphy.channel.ideal import generateRayleighChannel
from wiphy.code import generateCodes
from wiphy.util.general import getXORtoErrorBitsArray, inv_dB, randn_c, argToDic, saveCSV, normsYHCodes


def simulateBERReference(codes, channelfun, params, printValue=True):
    """
    Simulates BER values at multiple SNRs, where the straightforward reference algorithm is used. Note that this time complexity is unrealistically high. This simulation relies on the coherent maximum likelihood detector, that assumes perfect channel state information at the receiver. The environment variable USECUPY determines whether to use cupy or not.

    Args:
        codes (ndarray): an input codebook, which is generated on the CPU memory and is transferred into the GPU memory.
        channelfun (function): .
        params (dict): simulation parameters.
        printValue (bool): a flag that determines whether to print the simulated values.

    Returns:
        dict: a dict that has two keys: snr_dB and ber, and contains the corresponding results. All the results are transferred into the CPU memory.
    """
    IT, M, N, Nc, B = params["IT"], params["M"], params["N"], codes.shape[0], log2(codes.shape[0])
    snr_dBs = linspace(params["from"], params["to"], params["len"])
    sigmav2s = 1.0 / inv_dB(snr_dBs)
    xor2ebits = getXORtoErrorBitsArray(Nc)

    bers = zeros(params["len"])
    for i in trange(params["len"]):
        errorBits = 0
        v0 = randn_c(N, M) * sqrt(sigmav2s[i])  # N \times M
        s0 = eye(M, dtype=complex)
        for _ in range(IT):
            codei = random.randint(0, Nc)
            s1 = matmul(s0, codes[codei])  # differential encoding

            h = channelfun(N, M)  # N \times M
            v1 = randn_c(N, M) * sqrt(sigmav2s[i])  # N \times M

            y0 = matmul(h, s0) + v0  # N \times M
            y1 = matmul(h, s1) + v1  # N \times M

            # non-coherent detection that is free from the channel matrix h
            norms = normsYHCodes(y1, y0, codes)
            mini = argmin(norms)
            errorBits += xor2ebits[codei ^ mini].sum()

            v0 = v1
            s0 = s1

        bers[i] = errorBits / (IT * B)
        if printValue:
            print("At SNR = %1.2f dB, BER = %d / %d = %1.10e" % (snr_dBs[i], errorBits, IT * B, bers[i]))

    return {"snr_dB": snr_dBs, "ber": bers}


if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = sys.argv[1:]
    else:
        args = [
            "BER_sim=diff_channel=rayleigh_code=symbol_M=1_N=2_T=1_L=2_mod=PSK_IT=1e4_from=-10.00_to=30.00_len=21",
            "BER_sim=diff_channel=rayleigh_code=OSTBC_M=2_N=2_T=2_L=2_mod=PSK_IT=1e4_from=-10.00_to=30.00_len=21"
        ]

    for arg in args:
        print("Simulating arg = " + arg)
        params = argToDic(arg)
        codes = generateCodes(params)
        ret = simulateBERReference(codes, generateRayleighChannel, params, printValue=False)
        saveCSV(arg, ret)
        print(ret)
