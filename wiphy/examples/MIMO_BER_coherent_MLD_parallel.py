# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import sys
from numpy import *
from tqdm import trange
from wiphy.channel.ideal import generateAWGNChannel, generateRayleighChannel
from wiphy.code import generateCodes
from wiphy.util.general import getXORtoErrorBitsArray, inv_dB, randn_c, argToDic, saveCSV


def simulateBERParallel(codes, channelfun, params, printValue=True):
    """
    Simulates BER values at multiple SNRs, where the massively parallel algorithm is used. This implementation is especially designed for cupy.

    Args:
        codes (ndarray): an input codebook, which is generated on the CPU memory and is transferred into the GPU memory.
        channelfun (function): .
        params (dict): simulation parameters.
        printValue (bool): a flag that determines whether to print the simulated values.

    Returns:
        dict: a dict that has two keys: snr_dB and ber, and contains the corresponding results. All the results are transferred into the CPU memory.
    """

    M, N, T, ITo, ITi, Nc, B = params["M"], params["N"], params["T"], params["ITo"], params["ITi"], codes.shape[
        0], log2(codes.shape[0])
    snr_dBs = linspace(params["from"], params["to"], params["len"])
    sigmav2s = 1.0 / inv_dB(snr_dBs)
    codei = tile(arange(Nc), ITi)
    xor2ebits = getXORtoErrorBitsArray(Nc)

    x = hstack(tile(codes, Nc))  # M \times T * Nc^2
    # x = [codes[0] codes[0] ... codes[0] codes[1] ...]
    y = tile(hstack(codes), Nc)  # M \times T * Nc^2
    # y = [codes[0] codes[1] ... codes[Nc-1] codes[0] ...]
    diffxy = x - y  # M \times T * Nc^2

    bers = zeros(len(snr_dBs))
    for ito in trange(ITo, disable = not printValue):
        bigh = channelfun(N, M, ITi)  # ITi * N \times M
        bigv = tile(randn_c(ITi * N, T), Nc * Nc)  # ITi * N \times T * Nc^2

        for i in range(len(snr_dBs)):
            ydiff = matmul(bigh, diffxy) + bigv * sqrt(sigmav2s[i])  # ITi * N \times T * Nc^2
            ydifffro = square(abs(ydiff)).reshape(ITi, N, Nc * Nc, T)  # ITi \times N \times Nc * Nc \times T
            ydifffrosum = sum(ydifffro, axis=(1, 3))  # ITi \times Nc * Nc

            norms = ydifffrosum.reshape(ITi, Nc, Nc)  # ITi \times Nc \times Nc
            mini = argmin(norms, axis=2).reshape(ITi * Nc)
            errorBits = sum(xor2ebits[codei ^ mini])

            bers[i] += errorBits
            if printValue:
                nbits = (ito + 1) * ITi * B * Nc
                print("At SNR = %1.2f dB, BER = %d / %d = %1.10e" % (snr_dBs[i], bers[i], nbits, bers[i] / nbits))

    bers /= ITo * ITi * B * Nc
    return {"snr_dB": snr_dBs, "ber": bers}


if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = sys.argv[1:]
    else:
        args = [
            "BERP_sim=coh_channel=rayleigh_code=blast_M=4_T=1_L=2_mod=PSK_N=4_ITo=1e2_ITi=1e2_from=0.00_to=20.00_len=11",
            "BERP_sim=coh_channel=rayleigh_code=index_dm=dic_M=4_T=1_K=1_Q=4_L=4_mod=PSK_N=4_ITo=1e2_ITi=1e2_from=0.00_to=20.00_len=11"
        ]

    for arg in args:
        print("Simulating arg = " + arg)
        params = argToDic(arg)
        codes = asarray(generateCodes(params))
        ret = simulateBERParallel(codes, generateRayleighChannel, params, printValue=False)
        saveCSV(arg, ret)
        print(ret)
