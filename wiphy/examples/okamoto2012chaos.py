# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt
#
# The following code is an attempt to reproduce the papers [okamoto2012chaos] and [okamoto2016sto].
# Please note that we cannot guarantee that our implementation is correct.
# [okamoto2012chaos] E. Okamoto, ``A chaos MIMO transmission scheme for channel coding and physical-layer security,''
# IEICE Transactions on Communications, vol. E95-T, no. 4, 2012.
# [okamoto2016sto] E. Okamoto and N. Horiike, ``Performance improvement of chaos MIMO scheme using advanced stochastic characteristics,''
# IEICE Communications Express, vol. 1, 2016.

import sys

from numba import jit
from numpy import *
from numpy.testing import *
from tqdm import trange
from tqdm.auto import tqdm, trange
from wiphy.util.general import *


def eq2(b, a, m):
    if b[m] == 0:
        return a
    else:
        if a > .5:
            return 1. - a
        else:
            return a + 0.5


def bernoulli(x0, Ite):
    xs = zeros(Ite + 2)
    xs[0] = x0

    for i in range(1, Ite + 2):
        # mod 1 operation
        x = 2 * xs[i - 1]
        y = 1. - 1e-16
        xs[i] = x - y * int(x / y)

    return xs


def getOkamoto2016Symbols(b, M, T, Ite, c0):
    """
    Args:
        b: [0,1] M * T bits array
        M: the number of transmit antenna
        T: the number of time-slots
        Ite: the number of chaos iterations, e.g., 100

    Returns: Gaussian-distributed symbols s having length = M * T

    """
    MT = M * T
    c = zeros(MT + 1, dtype=complex)
    c[0] = c0
    s = zeros(MT, dtype=complex)

    for i in range(1, MT + 1):
        # print("i = " + str(i))
        x0r = eq2(b, a=real(c[i - 1]), m=i - 1)
        x0i = eq2(b, a=imag(c[i - 1]), m=i % MT)
        xsr = bernoulli(x0=x0r, Ite=Ite)
        xsi = bernoulli(x0=x0i, Ite=Ite)

        ci0r = xsr[Ite + b[int(i + MT / 2) % MT]]
        ci0i = xsi[Ite + b[int(i + MT / 2 + 1) % MT]]
        c[i] = ci0r + 1.j * ci0i

        cix = arccos(cos(37. * pi * (ci0r + ci0i))) / pi
        ciy = arcsin(sin(43. * pi * (ci0r - ci0i))) / pi + 0.5

        s[i - 1] = sqrt(-log(cix)) * (cos(2. * pi * ciy) + 1.j * sin(2. * pi * ciy))

    return s


def generateChaosMIMOCodes(M, T, Ite, c0):
    MT = M * T
    Nc = 2 ** (MT)

    codes = zeros((Nc, M, T), dtype=complex)
    for i in range(Nc):
        bits = convertIntToBinArray(i, M * T)
        s = getOkamoto2016Symbols(bits, M, T, Ite, c0)
        codes[i] = s.reshape(M, T) / sqrt(M)

    return codes


def testPowerConstraint(M, T, Ite):
    b = random.randint(2, size=M * T)
    c0 = random.uniform() + 1.j * random.uniform()
    s = getOkamoto2016Symbols(b, M, T, Ite, c0)

    assert_almost_equal(mean(real(s)), 0., decimal=2)
    assert_almost_equal(mean(imag(s)), 0., decimal=2)
    assert_almost_equal(mean(square(abs(s))), 1., decimal=2)

    import matplotlib.pyplot as plt
    plt.hist(real(s), bins=50, density=True, alpha=.5)
    plt.hist(imag(s), bins=50, density=True, alpha=.5)
    plt.show()


def showConstellation(M, T, Ite):
    Nc = 2 ** (M * T)

    import matplotlib.pyplot as plt
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    for i in trange(Nc):
        b = convertIntToBinArray(i, M * T)
        c0 = random.uniform() + 1.j * random.uniform()
        s = getOkamoto2016Symbols(b, M, T, Ite, c0)
        plt.plot(real(s), imag(s))

    plt.show()


def simulateBER(params):
    M, N, T, IT = params["M"], params["N"], params["T"], params["IT"]
    ITi = 100
    ITo = int(IT / ITi)
    MT = M * T
    Nc = 2 ** (MT)
    B = log2(Nc)

    snr_dBs = linspace(params["from"], params["to"], params["len"])
    sigmav2s = 1.0 / inv_dB(snr_dBs)
    xor2ebits = getXORtoErrorBitsArray(Nc)

    bers = zeros(len(snr_dBs))
    for i in range(len(snr_dBs)):
        errorBits = 0
        for ito in trange(ITo):
            c00 = random.uniform() + 1.j * random.uniform()
            codes = generateChaosMIMOCodes(M, T, 100, c00)

            for iti in range(ITi):
                codei = random.randint(0, Nc)

                H = randn_c(N, M)  # N \times M
                V = randn_c(N, T) * sqrt(sigmav2s[i])  # N \times T
                S = codes[codei]
                Y = H @ S + V  # N \times T

                p = square(abs(Y - H @ codes))  # Nc \times N \times T
                norms = sum(p, axis=(1, 2))  # summation over the (N,T) axes
                mini = argmin(norms)

                errorBits += sum(xor2ebits[codei ^ mini])

        bers[i] = errorBits / (IT * B)
        tqdm.write("At SNR = %1.2f dB, BER = %d / %d = %1.10e" %
                   (snr_dBs[i], errorBits, IT * B, bers[i]))

    return {"snr_dB": snr_dBs, "ber": bers}


def simulateBERKeyErr(params):
    M, N, T, IT = params["M"], params["N"], params["T"], params["IT"]
    ITi = 100
    ITo = int(IT / ITi)
    MT = M * T
    Nc = 2 ** (MT)
    B = log2(Nc)

    snr_dBs = linspace(params["from"], params["to"], params["len"])
    sigmav2s = 1.0 / inv_dB(snr_dBs)
    xor2ebits = getXORtoErrorBitsArray(Nc)

    bers = zeros(len(snr_dBs))
    for i in range(len(snr_dBs)):
        errorBits = 0
        for ito in trange(ITo):
            c00 = random.uniform() + 1.j * random.uniform()
            codes = generateChaosMIMOCodes(M, T, 100, c00)

            cdiff = random.uniform() + 1.j * random.uniform()
            c00b = c00 + cdiff * sqrt(sigmav2s[i] / 1e10)
            codesb = generateChaosMIMOCodes(M, T, 100, c00b)

            for iti in range(ITi):
                codei = random.randint(0, Nc)

                H = randn_c(N, M)  # N \times M
                V = randn_c(N, T) * sqrt(sigmav2s[i])  # N \times T
                S = codes[codei]
                Y = H @ S + V  # N \times T

                p = square(abs(Y - H @ codesb))  # Nc \times N \times T
                norms = sum(p, axis=(1, 2))  # summation over the (N,T) axes
                mini = argmin(norms)

                errorBits += sum(xor2ebits[codei ^ mini])

        bers[i] = errorBits / (IT * B)
        tqdm.write("At SNR = %1.2f dB, BER = %d / %d = %1.10e" %
                   (snr_dBs[i], errorBits, IT * B, bers[i]))

    return {"snr_dB": snr_dBs, "ber": bers}


@jit
def simulateSumOuter(Nc, codes, sigmav2, H, V):
    sum_outer = 0.0
    for outer in range(Nc):
        sum_inner = 0.0
        for inner in range(Nc):
            hxy = H @ (codes[outer] - codes[inner])
            head = hxy + V
            tail = V
            coeff = (-square(linalg.norm(head)) + square(linalg.norm(tail))) / sigmav2
            sum_inner += exp(coeff)
        sum_outer += log2(sum_inner)

    return sum_outer


def simulateAMI(params):
    M, N, T, IT = params["M"], params["N"], params["T"], params["IT"]
    ITi = 100
    ITo = int(IT / ITi)
    MT = M * T
    Nc = 2 ** (MT)
    B = log2(Nc)

    snr_dBs = linspace(params["from"], params["to"], params["len"])
    sigmav2s = 1.0 / inv_dB(snr_dBs)

    amis = zeros(len(snr_dBs))
    for i in range(len(snr_dBs)):
        sum_outer = 0.0
        for ito in trange(ITo):
            c00 = random.uniform() + 1.j * random.uniform()
            codes = generateChaosMIMOCodes(M, T, 100, c00)
            for iti in range(ITi):
                H = randn_c(N, M)  # N \times M
                V = randn_c(N, T) * sqrt(sigmav2s[i])  # N \times T
                sum_outer += simulateSumOuter(Nc, codes, sigmav2s[i], H, V)

        amis[i] = (B - sum_outer / Nc / (ITo * ITi)) / T
        tqdm.write("At SNR = %1.2f dB, BER = %1.10e" %
                   (snr_dBs[i], amis[i]))

    return {"snr_dB": snr_dBs, "ami": amis}


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        args.append("BERNOULLI")
        # args.append("POWER_M=1024_T=64_Ite=100")
        # args.append("CONSTELLATION_M=2_T=2_Ite=100")
        # args.append("CONSTELLATION_M=4_T=1_Ite=100")
        # args.append(
        #     "BER_channel=rayleigh_code=CMIMO_"
        #     "M=4_N=4_T=1_IT=1e5_from=0.00_to=40.00_len=9")
        # args.append(
        #     "BER_channel=rayleigh_code=CMIMO_"
        #     "M=4_N=4_T=2_IT=1e5_from=0.00_to=40.00_len=9")
        # args.append(
        #     "BER_channel=rayleigh_code=CMIMO_"
        #     "M=4_N=4_T=1_IT=1e5_from=0.00_to=40.00_len=9_option=keyerr")
        # args.append(
        #     "AMI_channel=rayleigh_code=CMIMO_"
        #     "M=4_N=4_T=1_IT=1e3_from=-30.00_to=20.00_len=11")

    for arg in args:
        print(arg)
        params = argToDic(arg)

        if params["mode"] == "BERNOULLI":
            import matplotlib.pyplot as plt

            xs = bernoulli(0.21111111, int(1e7))
            plt.hist(xs, bins=20, density=True, alpha=.5)
            plt.show()

        if params["mode"] == "POWER":
            testPowerConstraint(
                M=params["M"], T=params["T"], Ite=params["Ite"])

        elif params["mode"] == "CONSTELLATION":
            showConstellation(
                M=params["M"], T=params["T"], Ite=params["Ite"])

        elif params["mode"] == "BER":
            if "option" not in params or params["option"] != "keyerr":
                ret = simulateBER(params)
            else:
                ret = simulateBERKeyErr(params)
            saveCSV(arg, ret)
            print(ret)

        elif params["mode"] == "AMI":
            ret = simulateAMI(params)
            saveCSV(arg, ret)
            print(ret)
