# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt
#
# The following code is an attempt to reproduce the paper [kaddoum2011csk].
# Please note that we cannot guarantee that our implementation is correct.
# [kaddoum2011csk] G. Kaddoum, M. Vu, and F. Gagnon, ``On the performance of chaos shift keying in MIMO communications systems,''
# IEEE Wireless Communications and Networking Conference, Cancun, Quintana Roo, Mexico, March 28-31, 2011.

import sys

import numpy as np
from numba import njit
from tqdm import trange
from wiphy.code import generateCodes
from wiphy.util.general import *


@njit
def getSecondChebyshevPolynomialSequence(x0, size):
    """x0 has to be within [0,1]."""
    xs = np.zeros(size)
    # xs[0] = 2 * np.random.rand() - 1
    xs[0] = 2 * x0 - 1
    for k in range(1, len(xs)):
        xs[k] = 1 - 2 * xs[k - 1] ** 2

    # normalization
    xs -= np.mean(xs)
    xs /= np.sqrt(np.var(xs))

    return xs


def checkPowerConstraint(arg):
    params = argToDic(arg)
    codes = generateCodes(params)
    Nc = codes.shape[0]
    xs = getSecondChebyshevPolynomialSequence(0.24, params["M"] * params["IT"])

    ist = params["M"] * it
    ien = params["M"] * (it + 1)
    norms = np.array(
        [np.linalg.norm(np.matmul(codes, np.diag(xs[ist:ien])), axis=(1, 2))
         for it in range(params["IT"])])
    print("Mean norm = " + str(np.mean(np.square(norms)) / params["T"]))


def simulateBER(params):
    # second-order Chebyshev polynomial function
    if "x0" in params:
        xs = getSecondChebyshevPolynomialSequence(
            params["x0"], params["M"] * params["IT"])
        if "d" in params:
            ys = getSecondChebyshevPolynomialSequence(
                params["x0"] + params["d"], params["M"] * params["IT"])
        else:
            ys = xs

    codes = generateCodes(params)
    Nc = codes.shape[0]
    B = np.log2(Nc)

    snr_dBs = np.linspace(params["from"], params["to"], params["len"])
    sigmav2s = 1.0 / inv_dB(snr_dBs)
    xor2ebits = getXORtoErrorBitsArray(Nc)

    bers = np.zeros(len(snr_dBs))
    for i in range(len(snr_dBs)):
        if "option" in params and params["option"] == "keyerr":
            hx = randn_c()
            x0 = np.exp(-np.square(np.abs(hx)))
            hy = hx + randn_c() * np.sqrt(sigmav2s[i] / 1e10)
            y0 = np.exp(-np.square(np.abs(hy)))
            xs = getSecondChebyshevPolynomialSequence(
                x0, params["M"] * params["IT"])
            ys = getSecondChebyshevPolynomialSequence(
                y0, params["M"] * params["IT"])

        errorBits = 0
        for it in trange(params["IT"]):
            codei = np.random.randint(0, Nc)
            ist = params["M"] * it
            ien = params["M"] * (it + 1)
            px = np.diag(xs[ist: ien])
            py = np.diag(ys[ist: ien])

            h = randn_c(params["N"], params["M"])  # N \times M
            v = randn_c(params["N"], params["T"]) \
                * np.sqrt(sigmav2s[i])  # N \times T
            s = np.matmul(px, codes[codei])
            y = np.matmul(h, s) + v  # N \times T

            p = np.square(np.abs(y - np.matmul(h, np.matmul(py, codes))))  # Nc \times N \times T
            norms = np.sum(p, axis=(1, 2))  # summation over the (N,T) axes
            mini = np.argmin(norms)

            errorBits += np.sum(xor2ebits[codei ^ mini])

        bers[i] = errorBits / (params["IT"] * B)
        print("At SNR = %1.2f dB, BER = %d / %d = %1.10e" % (snr_dBs[i], errorBits, params["IT"] * B, bers[i]))

    return {"snr_dB": snr_dBs, "ber": bers}


if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    args = sys.argv[1:]
    if len(args) == 0:
        args.append(
            "BER_channel=rayleigh_code=OSTBC_O=2_L=256_mod=QAM_"
            "x0=0.24_M=4_N=4_T=4_IT=1e3_from=-10.00_to=40.00_len=26")
        args.append(
            "BER_channel=rayleigh_code=OSTBC_O=2_L=256_mod=QAM_"
            "M=4_N=4_T=4_IT=1e3_from=-10.00_to=40.00_len=26_option=keyerr")

    for arg in args:
        print(arg)
        params = argToDic(arg)
        if params["mode"] == "BER":
            ret = simulateBER(params)
            saveCSV(arg, ret)
            print(ret)
        elif params["mode"] == "CONST":
            M = params["M"]
            MT = params["M"] * params["T"]
            xs = getSecondChebyshevPolynomialSequence(params["x0"], M * params["IT"])
            codes = generateCodes(params)
            print(codes)
            Nc = codes.shape[0]

            symbols = np.zeros((MT * params["IT"]), dtype=np.complex)
            for it in range(params["IT"]):
                codei = np.random.randint(0, Nc)
                px = np.diag(xs[(M * it): (M * (it + 1))])
                symbols[(MT * it): (MT * (it + 1))] = np.matmul(px, codes[codei]).reshape(-1)

            df = {"real": np.real(symbols), "imag": np.imag(symbols)}
            saveCSV(arg, df)
