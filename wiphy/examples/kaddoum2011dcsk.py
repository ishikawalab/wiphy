# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt
#
# The following code is an attempt to reproduce the paper [kaddoum2011csk].
# Please note that we cannot guarantee that our implementation is correct.
# [kaddoum2011dcsk] G. Kaddoum, M. Vu, and F. Gagnon, ``Performance analysis of differential chaotic shift keying communications in MIMO systems,''
# IEEE International Symposium on Circuits and Systems, Rio de Janeiro, Brazil, May 15-18, 2011.

import sys

from numba import njit
from numpy import *
from tqdm import trange
from wiphy.code import generateCodes
from wiphy.util.general import *


@njit
def getSecondChebyshevPolynomialSequence(x0, size):
    """x0 has to be within [0,1]."""
    xs = zeros(size)
    # xs[0] = 2 * random.rand() - 1
    xs[0] = 2 * x0 - 1
    for k in range(1, len(xs)):
        xs[k] = 1 - 2 * xs[k - 1] ** 2

    # normalization
    xs -= mean(xs)
    xs /= sqrt(var(xs))

    return xs


def simulateBER(params):
    M, N, T, IT = params["M"], params["N"], params["T"], params["IT"]

    # second-order Chebyshev polynomial function
    if "x0" in params:
        x0 = params["x0"]
        xs = getSecondChebyshevPolynomialSequence(x0, M * IT)
        if "d" in params:
            d = params["d"]
            ys = getSecondChebyshevPolynomialSequence(x0 + d, M * IT)
        else:
            ys = xs

    codes = generateCodes(params)
    Nc = codes.shape[0]
    B = log2(Nc)

    snr_dBs = linspace(params["from"], params["to"], params["len"])
    sigmav2s = 1.0 / inv_dB(snr_dBs)
    xor2ebits = getXORtoErrorBitsArray(Nc)

    bers = zeros(len(snr_dBs))
    for i in range(len(snr_dBs)):
        if "option" in params and params["option"] == "keyerr":
            hx = randn_c()
            x0 = exp(-square(abs(hx)))
            hy = hx + randn_c() * sqrt(sigmav2s[i])
            y0 = exp(-square(abs(hy)))
            xs = getSecondChebyshevPolynomialSequence(x0, M * IT)
            ys = getSecondChebyshevPolynomialSequence(y0, M * IT)

        errorBits = 0
        s0 = eye(M, dtype=complex)
        px0 = eye(M, dtype=complex)
        py0 = eye(M, dtype=complex)
        for it in trange(IT):
            codei = random.randint(0, Nc)
            px1 = diag(xs[(M * it): M * (it + 1)])
            py1 = diag(ys[(M * it): M * (it + 1)])

            h = randn_c(N, M)  # N \times M
            v0 = randn_c(N, T) * sqrt(sigmav2s[i])  # N \times T
            v1 = randn_c(N, T) * sqrt(sigmav2s[i])  # N \times T
            s1 = matmul(s0, codes[codei])
            y0 = matmul(h, matmul(s0, px0)) + v0  # N \times T
            y1 = matmul(h, matmul(s1, px1)) + v1  # N \times T

            y0p = matmul(y0, linalg.inv(py0))
            p = square(abs(y1 - matmul(matmul(y0p, codes), py1)))  # Nc \times N \times T
            norms = sum(p, axis=(1, 2))  # summation over the (N,T) axes
            mini = argmin(norms)
            errorBits += sum(xor2ebits[codei ^ mini])

            s0 = s1
            px0 = px1
            py0 = py1

        bers[i] = errorBits / (IT * B)
        print("At SNR = %1.2f dB, BER = %d / %d = %1.10e" % (snr_dBs[i], errorBits, IT * B, bers[i]))

    return {"snr_dB": snr_dBs, "ber": bers}


if __name__ == '__main__':
    set_printoptions(linewidth=inf)
    args = sys.argv[1:]
    if len(args) == 0:
        args.append(
            "BER_channel=rayleigh_code=OSTBC_L=2_mod=PSK_"
            "x0=0.24_M=2_N=2_T=2_IT=1e3_from=-10.00_to=40.00_len=26")
        args.append(
            "BER_channel=rayleigh_code=OSTBC_O=2_L=256_mod=PSK_"
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

            symbols = zeros((MT * params["IT"]), dtype=complex)
            for it in range(params["IT"]):
                codei = random.randint(0, Nc)
                px = diag(xs[(M * it): (M * (it + 1))])
                symbols[(MT * it): (MT * (it + 1))] = matmul(px, codes[codei]).reshape(-1)

            df = {"real": real(symbols), "imag": imag(symbols)}
            saveCSV(arg, df)
