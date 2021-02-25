# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import sys
from numpy import *
from tqdm import trange
from wiphy.code import generateCodes
from wiphy.code.basis import generateBases
from wiphy.util.general import inv_dB, randn_c, argToDic, saveCSV, normsYHCodes, getErrorBitsTable


def simulateBERReference(codes, bases, alpha, IT, M, N, T, W, snr_dBs, printValue=True):
    """
    Simulates BER values at multiple SNRs, where the straightforward reference algorithm is used. Note that this time complexity is unrealistically high. This simulation relies on the nonsquare differential space-time block codes, which are proposed in [1-3]. This implementation uses the square-to-nonsquare projection concept of [2] and the adaptive forgetting factor of [3] for time-varying channels. The environment variable USECUPY determines whether to use cupy or not.

    - [1] N. Ishikawa and S. Sugiura, "Rectangular differential spatial modulation for open-loop noncoherent massive-MIMO downlink," IEEE Trans. Wirel. Commun., vol. 16, no. 3, pp. 1908--1920, 2017.
    - [2] N. Ishikawa, R. Rajashekar, C. Xu, S. Sugiura, and L. Hanzo, "Differential space-time coding dispensing with channel-estimation approaches the performance of its coherent counterpart in the open-loop massive MIMO-OFDM downlink," IEEE Trans. Commun., vol. 66, no. 12, pp. 6190--6204, 2018.
    - [3] N. Ishikawa, R. Rajashekar, C. Xu, M. El-Hajjar, S. Sugiura, L. L. Yang, and L. Hanzo, "Differential-detection aided large-scale generalized spatial modulation is capable of operating in high-mobility millimeter-wave channels," IEEE J. Sel. Top. Signal Process., in press.

    Args:
        codes (ndarray): an input codebook, which is generated on the CPU memory and is transferred into the GPU memory.
        bases (ndarray): a set of bases that transform a square matrix into a nonsquare matrix.
        channelfun (function): .
        params (dict): simulation parameters.
        printValue (bool): a flag that determines whether to print the simulated values.

    Returns:
        ret (dict): a dict that has two keys: snr_dB and ber, and contains the corresponding results. All the results are transferred into the CPU memory.
    """
    Nc, B = codes.shape[0], log2(codes.shape[0])
    sigmav2s = 1.0 / inv_dB(snr_dBs)
    xor2ebits = getErrorBitsTable(Nc)
    E1 = bases[0]  # M \times T
    E1H = E1.T.conj()
    Xrs = zeros((Nc, M, T)) + 0.j
    for x in range(Nc):
        Xrs[x] = codes[x] @ E1
    #Xrs = matmulb(codes, E1)  # Nc \times M \times Tc

    bers = zeros(len(snr_dBs))
    for i in trange(len(snr_dBs)):
        errorBits = 0

        for it in range(IT):
            S0 = eye(M) + 0.j
            Yhat0 = Yhat1 = zeros((N, M)) + 0.j

            H = randn_c(N, M)  # N \times M

            for wi in range(1, int(W / T) + 1):
                if wi <= M / T:
                    S1 = eye(M) + 0.j
                    Sr1 = bases[wi - 1]
                    X1 = S1
                    Y1 = H @ Sr1 + randn_c(N, T) * sqrt(sigmav2s[i])  # N \times T
                    Yhat1 += Y1 @ bases[wi - 1].T.conj()
                else:
                    codei = random.randint(0, Nc)
                    X1 = codes[codei]
                    S1 = S0 @ X1
                    Sr1 = S1 @ E1
                    Y1 = H @ Sr1 + randn_c(N, T) * sqrt(sigmav2s[i])  # N \times T

                    # estimate
                    norms = normsYHCodes(Y1, Yhat0, Xrs)
                    mini = argmin(norms)
                    Xhat1 = codes[mini]

                    Yhd = Yhat0 @ Xhat1
                    D1 = Y1 - Yhd @ E1
                    Yhat1 = (1.0 - alpha) * D1 @ E1H + Yhd

                    # # adaptive forgetting factor for time-varying channels
                    # Yhd = Yhat0 @ Xhat1
                    # D1 = Y1 - Yhd @ E1
                    # n1 = square(linalg.norm(D1))
                    # estimatedAlpha = N * T * sigmav2s[i] / n1
                    # estimatedAlpha = min(max(estimatedAlpha, 0.01), 0.99)
                    # Yhat1 = (1.0 - estimatedAlpha) * D1 @ E1H + Yhd

                    errorBits += xor2ebits[codei][mini]

                S0 = S1
                Yhat0 = Yhat1

        bers[i] = errorBits / (IT * B * (W - M)) * T

        if printValue:
            print("At SNR = %1.2f dB, BER = %d / %d = %1.20e" % (
                snr_dBs[i], errorBits, (IT * B * (W - M)) / T, bers[i]))

    return {"snr_dB": snr_dBs, "ber": bers}


if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = sys.argv[1:]
    else:
        args = [
            # (M, N, R) = (4, 4, 4.0)
            ## the N-DUC scheme using the dense basis
            "BER_sim=nsdiff_channel=rayleigh_code=DUC_basis=d_alpha=0.810_M=4_N=4_T=1_L=16_W=80_IT=1e3_from=0.00_to=30.00_len=16",
            ## the N-ADSM scheme using the sparse basis
            "BER_sim=nsdiff_channel=rayleigh_code=ADSM_basis=i_alpha=0.797_M=4_N=4_T=1_L=4_mod=PSK_W=80_IT=1e3_from=0.00_to=30.00_len=16"
        ]

    for arg in args:
        print("Simulating arg = " + arg)
        params = argToDic(arg)
        codes = generateCodes(params)
        bases = generateBases(params["basis"], params["M"], params["T"])
        snr_dBs = linspace(params["from"], params["to"], params["len"])
        ret = simulateBERReference(codes, bases, params["alpha"], params["IT"], params["M"], params["N"], params["T"], params["W"], snr_dBs, printValue=False)
        saveCSV(arg, ret)
        print(ret)
