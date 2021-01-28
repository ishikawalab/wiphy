# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt
# USECUPY = 0 required

import sys
from numpy import *
from tqdm import trange

from wiphy.channel.ideal import generateRayleighChannel
from wiphy.code import generateCodes
from wiphy.util.general import getXORtoErrorBitsArray, inv_dB, randn_c, argToDic, dicToNumpy, saveCSV


def simulateBERReference(codes, channelfun, params, printValue=True):
    """
    Simulates BER values at multiple SNRs, where the straightforward reference algorithm is used. Note that this time complexity is unrealistically high. This simulation relies on the coherent maximum likelihood detector, that assumes perfect channel state information at the receiver. The environment variable USECUPY determines whether to use cupy or not.

    Args:
        codes (ndarray): an input codebook, which is generated on the CPU memory and is transferred into the GPU memory.
        channelfun (function): .
        params (dict): simulation parameters.
        outputFile (bool): a flag that determines whether to output the obtained results to the results/ directory.
        printValue (bool): a flag that determines whether to print the simulated values.

    Returns:
        dict: a dict that has two keys: snr_dB and ber, and contains the corresponding results. All the results are transferred into the CPU memory.
    """

    IT, M, N, T, Nc, B = params["IT"], params["M"], params["N"], params["T"], codes.shape[0], log2(codes.shape[0])
    snr_dBs = linspace(params["from"], params["to"], params["len"])
    sigmav2s = 1.0 / inv_dB(snr_dBs)
    xor2ebits = getXORtoErrorBitsArray(Nc)

    bers = zeros(len(snr_dBs))
    for i in trange(len(snr_dBs)):
        errorBits = 0
        for _ in range(IT):
            codei = random.randint(0, Nc)
            H = channelfun(N, M)  # N \times M
            V = randn_c(N, T) * sqrt(sigmav2s[i])  # N \times T
            Y = matmul(H, codes[codei]) + V  # N \times T

            p = square(abs(Y - matmul(H, codes)))  # Nc \times N \times T
            norms = sum(p, axis=(1, 2))  # summation over the (N,T) axes
            mini = argmin(norms)
            errorBits += sum(xor2ebits[codei ^ mini])

        bers[i] = errorBits / (IT * B)
        if printValue:
            print("At SNR = %1.2f dB, BER = %d / %d = %1.10e" % (snr_dBs[i], errorBits, IT * B, bers[i]))

    return dicToNumpy({"snr_dB": snr_dBs, "ber": bers})


if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = sys.argv[1:]
    else:
        args = [
            "BER_sim=coh_channel=rayleigh_code=blast_M=4_T=1_L=2_mod=PSK_N=4_IT=1e4_from=0.00_to=20.00_len=11",
            "BER_sim=coh_channel=rayleigh_code=index_dm=dic_M=4_T=1_K=1_Q=4_L=4_mod=PSK_N=4_IT=1e4_from=0.00_to=20.00_len=11"
        ]

    for arg in args:
        print("Simulating arg = " + arg)
        params = argToDic(arg)
        codes = generateCodes(params)
        ret = simulateBERReference(codes, generateRayleighChannel, params, printValue=False)
        saveCSV(arg, ret)
        print(ret)
