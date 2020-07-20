# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt
# USECUPY = 0 required

from numpy import *
from tqdm import trange

from wiphy.channel.ideal import generateRayleighChannel
from wiphy.code import generateCodes
from wiphy.util.general import inv_dB, randn_c, argToDic, dicToNumpy, saveCSV


def simulateAMIReference(codes, channelfun, params, printValue=True):
    """
    Simulates AMI values at multiple SNRs, where the straightforward reference algorithm is used. Note that this time complexity is unrealistically high.

    Args:
        codes (ndarray): an input codebook, which is generated on the CPU memory and is transferred into the GPU memory.
        channelfun (function): .
        params (dict): simulation parameters.
        outputFile (bool): a flag that determines whether to output the obtained results to the results/ directory.
        printValue (bool): a flag that determines whether to print the simulated values.

    Returns:
        dict: a dict that has two keys: snr_dB and ami, and contains the corresponding results. All the results are transferred into the CPU memory.
    """

    IT, M, N, T, Nc, B = params["IT"], params["M"], params["N"], params["T"], codes.shape[0], log2(codes.shape[0])
    snr_dBs = linspace(params["from"], params["to"], params["len"])
    sigmav2s = 1.0 / inv_dB(snr_dBs)

    amis = zeros(len(snr_dBs))
    for i in trange(len(snr_dBs)):
        sum_outer = 0.0
        for _ in range(IT):
            V = sqrt(sigmav2s[i]) * randn_c(N, T)
            H = channelfun(N, M)  # N \times M
            for outer in range(Nc):
                sum_inner = 0.0
                for inner in range(Nc):
                    hxy = matmul(H, codes[outer] - codes[inner])
                    head = hxy + V
                    tail = V
                    coeff = (-square(linalg.norm(head)) + square(linalg.norm(tail))) / sigmav2s[i]
                    sum_inner += exp(coeff)
                sum_outer += log2(sum_inner)
        amis[i] = (B - sum_outer / Nc / IT) / T
        if printValue:
            print("At SNR = %1.2f dB, AMI = %1.10f" % (snr_dBs[i], amis[i]))

    return dicToNumpy({"snr_dB": snr_dBs, "ami": amis})


if __name__ == '__main__':
    args = ["AMI_sim=coh_channel=rayleigh_code=blast_M=4_T=1_L=2_mod=PSK_N=4_IT=1e3_from=0.00_to=20.00_len=11",
            "AMI_sim=coh_channel=rayleigh_code=index_dm=dic_M=4_T=1_K=1_Q=4_L=4_mod=PSK_N=4_IT=1e3_from=0.00_to=20.00_len=11"]

    for arg in args:
        print("Simulating arg = " + arg)
        params = argToDic(arg)
        codes = generateCodes(params)
        ret = simulateAMIReference(codes, generateRayleighChannel, params, printValue=False)
        saveCSV(arg, ret)
