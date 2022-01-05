# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import sys
from numpy import *
from tqdm import trange
from wiphy.channel.ideal import generateRayleighChannel
from wiphy.code import generateCodes
from wiphy.util.general import getXORtoErrorBitsArray, inv_dB, randn_c, argToDic, saveCSV


def simulateBERParallel(codes, channelfun, params, printValue=True):
    """
    Simulates BER values at multiple SNRs, where the massively parallel algorithm is used. This implementation is especially designed for cupy. This simulation relies on the coherent maximum likelihood detector, that assumes perfect channel state information at the receiver. The environment variable USECUPY determines whether to use cupy or not.

    Args:
        codes (ndarray): an input codebook, which is generated on the CPU memory and is transferred into the GPU memory.
        channelfun (function): .
        params (dict): simulation parameters.
        printValue (bool): a flag that determines whether to print the simulated values.

    Returns:
        dict: a dict that has two keys: snr_dB and ber, and contains the corresponding results. All the results are transferred into the CPU memory.
    """
    M, N, ITo, ITi, Nc, B = params["M"], params["N"], params["ITo"], params["ITi"], codes.shape[0], log2(codes.shape[0])

    if Nc > ITi:
        print("ITi should be larger than Nc = %d." % Nc)

    snr_dBs = linspace(params["from"], params["to"], params["len"])
    sigmav2s = 1.0 / inv_dB(snr_dBs)
    xor2ebits = getXORtoErrorBitsArray(Nc)
    codesmat = hstack(codes)  # M \times M * Nc
    eyes = tile(eye(M, dtype=complex), ITi).T.reshape(ITi, M, M)  # ITi \times M \times M

    indspermute = random.permutation(arange(ITi))
    codei = tile(arange(Nc), int(ceil(ITi / Nc)))[0:ITi]
    X1 = take(codes, codei, axis=0)  # ITi \times M \times M very slow
    V0 = randn_c(ITi, N, M)  # ITi \times N \times M
    S0 = eyes

    bers = zeros(len(snr_dBs))
    for ito in trange(ITo):
        H = channelfun(N, M, ITi).reshape(ITi, N, M)  # ITi \times N \times M
        V1 = randn_c(ITi, N, M)  # ITi \times N \times M
        S1 = matmul(S0, X1)

        for i in range(len(snr_dBs)):
            Y0 = matmul(H, S0) + V0 * sqrt(sigmav2s[i])  # ITi \times N \times M
            Y1 = matmul(H, S1) + V1 * sqrt(sigmav2s[i])  # ITi \times N \times M

            Y0X = matmul(Y0, codesmat)  # ITi \times N \times M * Nc
            Ydiff = tile(Y1, Nc) - Y0X  # ITi \times N \times M * Nc
            Ydifffro = square(abs(Ydiff)).reshape(ITi, N, Nc, M)  # ITi \times N \times Nc \times M
            norms = sum(Ydifffro, axis=(1, 3))  # ITi \times Nc
            mini = argmin(norms, axis=1)  # ITi

            errorBits = sum(xor2ebits[codei ^ mini])
            bers[i] += errorBits
            nbits = (ito + 1) * ITi * B
            if printValue:
                print("At SNR = %1.2f dB, BER = %d / %d = %1.10e" % (snr_dBs[i], bers[i], nbits, bers[i] / nbits))

        V0 = V1
        S0 = S1
        codei = codei[indspermute]
        X1 = X1[indspermute]

    bers /= ITo * ITi * B
    return {"snr_dB": snr_dBs, "ber": bers}


if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = sys.argv[1:]
    else:
        args = [
            "BERP_sim=diff_channel=rayleigh_code=symbol_M=1_N=2_T=1_L=2_mod=PSK_ITo=1e1_ITi=1e3_from=-10.00_to=30.00_len=21",
            "BERP_sim=diff_channel=rayleigh_code=OSTBC_M=2_N=2_T=2_L=2_mod=PSK_ITo=1e1_ITi=1e3_from=-10.00_to=30.00_len=21",
            "BERP_sim=diff_channel=rayleigh_code=adsm_M=4_T=4_O=2_L=4_mod=PSK_N=4_ITo=1e2_ITi=1e4_from=0.00_to=10.00_len=11"
        ]

    for arg in args:
        print("Simulating arg = " + arg)
        params = argToDic(arg)
        codes = asarray(generateCodes(params))
        ret = simulateBERParallel(codes, generateRayleighChannel, params, printValue=False)
        saveCSV(arg, ret)
        print(ret)
