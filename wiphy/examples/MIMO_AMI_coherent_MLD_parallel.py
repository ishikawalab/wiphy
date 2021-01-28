# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import sys
from numpy import *
from tqdm import trange
from wiphy.channel.ideal import generateRayleighChannel
from wiphy.code import generateCodes
from wiphy.util.general import inv_dB, randn_c, argToDic, saveCSV


def simulateAMIParallel(codes, channelfun, params, printValue=True):
    """
    Simulates AMI values at multiple SNRs, where the massively parallel algorithm is used. This implementation is especially designed for cupy.

    Args:
        codes (ndarray): an input codebook, which is generated on the CPU memory and is transferred into the GPU memory.
        channelfun (function): .
        params (dict): simulation parameters.
        printValue (bool): a flag that determines whether to print the simulated values.

    Returns:
        dict: a dict that has two keys: snr_dB and ami, and contains the corresponding results. All the results are transferred into the CPU memory.
    """

    M, N, T, ITo, ITi, Nc, B = params["M"], params["N"], params["T"], params["ITo"], params["ITi"], codes.shape[
        0], log2(codes.shape[0])
    snr_dBs = linspace(params["from"], params["to"], params["len"])
    sigmav2s = 1.0 / inv_dB(snr_dBs)

    # The following three variables are the same as those used in simulateBERParallel
    x = hstack(tile(codes, Nc))  # M \times T * Nc^2
    y = tile(hstack(codes), Nc)  # M \times T * Nc^2
    diffxy = x - y  # M \times T * Nc^2

    amis = zeros(len(snr_dBs))
    for ito in trange(ITo):
        bigh = channelfun(N, M, ITi)  # ITi * N \times M
        bigv = tile(randn_c(ITi * N, T), Nc * Nc)  # ITi * N \times T * Nc^2

        bigvfro = square(abs(bigv)).reshape(ITi, N, Nc * Nc, T)  # ITi \times N \times Nc^2 \times T
        frov = sum(bigvfro, axis=(1, 3)).reshape(ITi, Nc, Nc)  # ITi \times Nc \times Nc

        for i in range(len(snr_dBs)):
            hsplusv = matmul(bigh, diffxy) + bigv * sqrt(sigmav2s[i])  # ITi * N \times T * Nc^2
            hsvfro = square(abs(hsplusv)).reshape(ITi, N, Nc * Nc, T)  # ITi \times N \times Nc^2 \times T
            froy = sum(hsvfro, axis=(1, 3))  # ITi \times Nc^2
            reds = froy.reshape(ITi, Nc, Nc)  # ITi \times Nc \times Nc

            ecoffs = -reds / sigmav2s[i] + frov  # diagonal elements must be zero
            bminus = mean(log2(sum(exp(ecoffs), axis=2)))

            amis[i] += (B - bminus) / T
            if printValue:
                print("At SNR = %1.2f dB, AMI = %1.10f" % (snr_dBs[i], amis[i] / (ito + 1)))

    #
    amis /= ITo
    return {"snr_dB": snr_dBs, "ami": amis}


if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = sys.argv[1:]
    else:
        args = [
            "AMIP_sim=coh_channel=rayleigh_code=blast_M=4_T=1_L=2_mod=PSK_N=4_ITo=1e1_ITi=1e2_from=0.00_to=20.00_len=11",
            "AMIP_sim=coh_channel=rayleigh_code=index_dm=dic_M=4_T=1_K=1_Q=4_L=4_mod=PSK_N=4_ITo=1e1_ITi=1e2_from=0.00_to=20.00_len=11"
        ]

    for arg in args:
        print("Simulating arg = " + arg)
        params = argToDic(arg)
        codes = asarray(generateCodes(params))
        ret = simulateAMIParallel(codes, generateRayleighChannel, params, printValue=False)
        saveCSV(arg, ret)
        print(ret)
