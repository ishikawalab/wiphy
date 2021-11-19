# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import sys
from timeit import timeit
import joblib
from numpy import *
from tqdm import trange
import matplotlib.pyplot as plt

from wiphy.channel.ideal import generateRayleighChannel
from wiphy.code import generateCodes
from wiphy.util.general import argToDic, saveCSV
from MIMO_AMI_coherent_MLD_parallel import simulateAMIParallel
from MIMO_BER_coherent_MLD_parallel import simulateBERParallel
from MIMO_BER_differential_MLD_reference import simulateBERReference


def simulateBERReferenceJoblib(codes, channelfun, params):
    snr_dBs = linspace(params["from"], params["to"], params["len"])
    listParams = []
    for snr_dB in snr_dBs:
        newParams = params.copy()
        newParams["from"] = newParams["to"] = snr_dB
        newParams["len"] = 1
        listParams.append(newParams)

    BERs = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(simulateBERReference)(codes = codes, channelfun = channelfun, params = p, printValue = False)
        for p in listParams
    )
    # print(BERs)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        pBLAST = argToDic("AMIP_sim=coh_channel=rayleigh_code=blast_M=4_T=1_L=2_mod=PSK_N=4_ITo=1e2_ITi=1e3_from=-10.00_to=30.00_len=21")
        cBLAST = asarray(generateCodes(pBLAST))

        pOSTBC = argToDic("BER_sim=diff_channel=rayleigh_code=OSTBC_M=2_N=2_T=2_L=2_mod=PSK_IT=1e5_from=-10.00_to=30.00_len=21")
        cOSTBC = asarray(generateCodes(pOSTBC))

        strFuncs = [
            "simulateAMIParallel(cBLAST, generateRayleighChannel, pBLAST, printValue=False)",
            "simulateBERParallel(cBLAST, generateRayleighChannel, pBLAST, printValue=False)",
            "simulateBERReference(cOSTBC, generateRayleighChannel, pOSTBC, printValue=False)",
            "simulateBERReferenceJoblib(cOSTBC, generateRayleighChannel, pOSTBC)"]

        meanTimes = zeros(len(strFuncs))
        for _ in trange(10):
            for i in range(len(strFuncs)):
                meanTimes[i] += timeit(strFuncs[i], globals=locals(), number=1)
        meanTimes /= 10.0
        print(meanTimes)

    else:
        timesi9 = [54.81692204, 45.23677923, 51.60407887,  7.59387721]
        timesr9 = [50.15030531, 39.24208482, 41.44064043,  6.43697599]
        timesm1 = [22.41968313, 19.87039665, 49.28366062, 10.89323277]

        labels = ["BLAST AMI\n(single/tensor)", "BLST BER\n(single/tensor)", "DOSTBC BER\n(single/ref.)", "DOSTBC BER\n(multi/ref.)"]
        left = arange(len(labels))
        width = 0.3

        plt.ylabel("Mean effective time [s]")
        plt.ylim(0, 60)

        plt.bar(left + 0 * width, timesi9, color='skyblue', edgecolor="black", hatch='/', width=width, align='center', label="Intel Core i9-10900K, 64GB memory")
        plt.bar(left + 1 * width, timesr9, color='tomato', edgecolor="black", hatch='\\', width=width, align='center', label="AMD Ryzen 9 5900HX, 32GB memory")
        plt.bar(left + 2 * width, timesm1, color='lightgreen', edgecolor="black", hatch='x', width=width, align='center', label="Apple M1, 8GB memory")

        plt.xticks(left + width, labels)
        plt.legend(loc='best')#, shadow=True)
        plt.show()




