# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

from timeit import timeit
import joblib
from numpy import *
from wiphy.channel.ideal import generateRayleighChannel
from wiphy.code import generateCodes
from wiphy.util.general import argToDic, saveCSV
from MIMO_AMI_coherent_MLD_parallel import simulateAMIParallel
from MIMO_BER_coherent_MLD_parallel import simulateBERParallel
from MIMO_BER_differential_MLD_reference import simulateBERReference

from tqdm import trange

def calc(var):
    lst = []
    for i in range(100):
        lst.append(i)

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
    #print(timeit.timeit("test()", globals = locals()))
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
            meanTimes[i] += timeit(strFuncs[i], globals=locals(), number = 1)
    meanTimes /= 10.0

    print(meanTimes)

    #print(cOSTBC)
    #print(repeat(cOSTBC, 5))

    #for i, j in zip(repeat(cOSTBC, 5), range(5)):
    #    print(i)

