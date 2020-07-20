# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

__all__ = ['generateBases', 'convertUnitaryToBases', 'constructUnitaryFromE1', 'getGSPE1']

import numpy as np

from ..util.general import getDFTMatrixNumpy, CayleyTransform, getRandomHermitianMatrix


def generateBases(type, M, T, **kwargs):
    """
    Generates a basis set for nonsquare differential encoding and decoding, which is proposed in [1,2].

    - [1] N. Ishikawa, R. Rajashekar, C. Xu, S. Sugiura, and L. Hanzo, ``Differential space-time coding dispensing with channel-estimation approaches the performance of its coherent counterpart in the open-loop massive MIMO-OFDM downlink,'' IEEE Trans. Commun., vol. 66, no. 12, pp. 6190–6204, 2018.
    - [2] N. Ishikawa, R. Rajashekar, C. Xu, M. El-Hajjar, S. Sugiura, L. L. Yang, and L. Hanzo, ``Differential-detection aided large-scale generalized spatial modulation is capable of operating in high-mobility millimeter-wave channels,'' IEEE J. Sel. Top. Signal Process., in press.

    Args:
        type (string): the basis type, such as i (IdentityBasis) and d (DFTBasis).
        M (int): the number of transmit antennas.
        T (int): the number of reduced time slots.

    Returns:
        numpy.ndarray: a set of bases, (M/T) \times M \times T.
    """
    # initialize a unitary matrix that generates a set of bases
    if type[0].lower() == 'i':
        # Identity basis
        U = np.eye(M, dtype=np.complex)
    elif type[0].lower() == 'd':
        # DFT basis
        U = getDFTMatrixNumpy(M)
    elif type[0].lower() == 'r':
        # Random basis
        U = CayleyTransform(getRandomHermitianMatrix(M))
    elif type[0].lower() == 'h':
        # Hybrid basis
        P = int(type.replace('h', ''))
        W = getDFTMatrixNumpy(P)
        U = np.zeros((M, M), dtype=complex)
        for i in range(int(M / P)):
            U[(i * P): (i * P + P), (i * P): (i * P + P)] = W
    elif type[0].lower() == 'g':
        # Gram–Schmidt Process basis
        E1 = kwargs["E1"] if "E1" in kwargs else None
        U = constructUnitaryFromE1(E1)

    return convertUnitaryToBases(U, T)  # (M/T) \times M \times T


def convertUnitaryToBases(U, T):
    M = U.shape[0]
    return np.array(np.hsplit(U, M / T))


def constructUnitaryFromE1(E1):
    M = E1.shape[0]
    T = E1.shape[1]
    U = np.zeros((M, M), dtype=np.complex)
    U[:, 0: T] = E1

    W = getDFTMatrixNumpy(M)
    for k in range(1, int(M / T)):
        v = W[:, (k * T): ((k + 1) * T)]
        msum = np.eye(M, dtype=np.complex)
        for i in range(k):
            E = U[:, (i * T): ((i + 1) * T)]
            msum -= np.matmul(E, E.T.conj())

        newE = np.matmul(msum, v)
        newE *= np.sqrt(T) / np.linalg.norm(newE)
        U[:, (k * T): ((k + 1) * T)] = newE

    return U


def getGSPE1(params):
    if params.code == "DUC":
        if params.M == 2 and params.L == 16:
            print("Basis.py: E1 designed for DUC, M=2, L=16, Powell MED = 0.5857878220964823")
            return np.array([[0.5227111661 - 0.0422474373j], [0.8507822376 - 0.0340260934j]])

        if params.M == 4 and params.L == 16:
            # print("Basis.py: E1 designed for DUC, M=4, L=16, MED = 1.9999999999999978")
            # return np.array([[0.5],[0.5],[0.5],[0.5]], dtype=np.complex)
            # print("Basis.py: E1 designed for DUC, M=4, L=16, Powell MED = 1.9999999918507845")
            # return np.array([[-0.1552578934 - 0.4752841151j],
            #      [0.4308312897 - 0.2537408069j],
            #      [0.4788382708 + 0.1439232794j],
            #      [-0.2577704955 - 0.4284324579j]])
            # print("Basis.py: E1 designed for DUC, M=4, L=16, [-19, 21] Powell MED = 1.909912484636119")
            # return np.array([[0.5303513637 - 0.1181744979j],
            #       [0.3900705105 + 0.3696020962j],
            #       [0.3630695691 + 0.3227808541j],
            #       [0.3398999781 + 0.2538947017j]])
            print("Basis.py: E1 designed for DUC, M=4, L=16, [0.5, 1.5] Powell MED = 1.9709200541641247")
            return np.array([[0.4114066914 - 0.2749088908j], [0.3337686103 - 0.3770792774j],
                             [0.4787497316 - 0.1444313845j], [0.3201948057 - 0.3859960174j]])

        if params.M == 4 and params.L == 256:
            print("Basis.py: E1 designed for DUC, M=4, L=256, Powell MED = 0.5664886704273715")
            return np.array([[0.5528659135 - 1.1061644806e-10j], [0.5308003283 - 5.3650351317e-09j],
                             [0.4531628012 - 2.3370365749e-02j], [0.4546290738 - 4.6739417600e-07j]])

        if params.M == 16 and params.L == 16:
            print("Basis.py: E1 designed for DUC, M=16, L=16, MED = 1.9924286420642512")
            return np.array([[-0.0721935097 - 0.2680306964j], [-0.0746096107 + 0.1028631665j],
                             [0.3238716415 + 0.1208974311j], [-0.1140521122 + 0.1472216356j],
                             [0.3400692302 - 0.0736020176j], [-0.0992137116 + 0.181828124j],
                             [-0.0949402647 + 0.2263729187j], [-0.136657163 - 0.0810585943j],
                             [-0.3187876384 - 0.0337700924j], [-0.2003138313 + 0.1358389316j],
                             [-0.2905808133 - 0.0242917252j], [-0.0054146435 - 0.0823672701j],
                             [0.2053385553 + 0.0274103506j], [0.043571457 - 0.220739553j],
                             [0.0290657709 + 0.2220697818j], [-0.2326738077 + 0.2282860375j]])

        if params.M == 16 and params.L == 64:
            print(
                "Basis.py: E1 designed for DUC, M=16, L=64")
            print(
                "[0.5, 1.5] Powell MED = 1.9290890209598217, mean multipoints")
            return np.array([[0.0924348279 - 0.2258516098j], [0.2471774435 + 0.0666293805j],
                             [0.1936904059 - 0.1600618383j], [0.2248012437 + 0.0932254522j],
                             [0.1363295714 - 0.2038251094j], [0.2000807088 - 0.156587474j],
                             [0.2495646509 + 0.0205364132j], [0.23834403 + 0.0741419057j],
                             [0.2327947437 - 0.0897079035j], [0.194560541 + 0.1513729679j],
                             [0.2172092145 + 0.1142378662j], [-0.193009212 + 0.1625447282j],
                             [0.2509817143 - 0.013757908j], [0.2007265938 - 0.1541232271j],
                             [-0.2535304175 - 0.0386404059j], [-0.2162803266 + 0.127164074j]])

        if params.M == 16 and params.L == 256:
            print("Basis.py: E1 designed for DUC, M=16, L=256, Powell MED = 1.4264979481928033")
            return np.array([[0.2463452635 - 3.3055051557e-04j], [0.159836629 - 5.2898928154e-08j],
                             [0.2720304184 + 9.6319527208e-05j], [0.2563415547 - 2.0555231402e-09j],
                             [0.2640806367 + 5.6309294056e-03j], [0.2592969927 - 2.7716153091e-02j],
                             [0.1360651104 - 1.7573466676e-08j], [0.2781522162 - 3.4816944581e-10j],
                             [0.2620909706 + 2.0935826459e-03j], [0.2336880281 - 2.3029082729e-07j],
                             [0.2592632202 - 2.2933490034e-06j], [0.2763768422 - 1.9740120331e-03j],
                             [0.2648882725 - 3.1630009520e-03j], [0.2624671215 + 1.1820720749e-03j],
                             [0.2571940053 + 1.9575075518e-04j], [0.2612843824 + 1.6397913134e-03j]])

        if params.M == 64 and params.L == 256:
            print("Basis.py: E1 designed for DUC, M=64, L=256, Powell MED = 1.7313571839396538")
            return np.array([[0.0447116806 - 0.1149321366j], [0.1411923202 - 0.0721136717j],
                             [0.1798119527 - 0.0758647504j], [-0.2020854509 + 0.040021071j],
                             [0.0973483605 - 0.1187560823j], [0.0787957316 - 0.0201386058j],
                             [0.1391940364 - 0.1335338442j], [0.1595426129 - 0.0545510855j],
                             [-0.0646271879 - 0.0907238296j], [0.0699048609 - 0.1279990456j],
                             [0.0459101937 - 0.0287385317j], [0.0488450468 + 0.0266190071j],
                             [-0.1453701377 - 0.0246436664j], [-0.114300942 - 0.0215822557j],
                             [0.0563526238 + 0.0866623003j], [0.0845172912 - 0.0951722167j],
                             [-0.0678150438 - 0.0674745258j], [0.0613942635 - 0.0602455558j],
                             [-0.041527009 + 0.0455002387j], [0.1096338125 - 0.1284948354j],
                             [0.0765861231 - 0.0673267087j], [0.0620325188 - 0.0620617232j],
                             [0.0478881662 - 0.1153195594j], [0.0819192645 + 0.1349508453j],
                             [0.1906939379 - 0.0866317378j], [0.1737728323 - 0.057913333j],
                             [0.0928173818 + 0.1139923173j], [0.0371203809 - 0.0609984747j],
                             [-0.1145703096 + 0.0812930461j], [0.0066896313 + 0.1251068253j],
                             [0.0207900386 - 0.0998733534j], [0.125902983 - 0.1250754368j],
                             [0.0753178841 - 0.0569505293j], [0.0659744921 + 0.0547523856j],
                             [0.0833475762 - 0.0984346522j], [-0.0105081255 - 0.0481012799j],
                             [0.0373526054 + 0.0963516259j], [0.0484246594 + 0.0662614288j],
                             [0.0540651183 - 0.0349246829j], [-0.0422478252 + 0.0385468822j],
                             [0.0358201317 - 0.054276622j], [0.0919028516 - 0.1105344319j],
                             [0.0449802267 + 0.0508738162j], [0.0810939707 + 0.0896410468j],
                             [0.0322947854 - 0.0568583282j], [0.0402384303 + 0.0957056326j],
                             [0.0166781958 + 0.0505806565j], [0.1588552084 + 0.0721977366j],
                             [0.0282647609 + 0.0793603996j], [-0.0657635827 - 0.0871395409j],
                             [0.0958914933 - 0.0911333898j], [0.0827958337 - 0.0043383603j],
                             [0.2017696251 - 0.0261526967j], [0.1098000104 + 0.0243674135j],
                             [-0.1285463763 - 0.0444430415j], [0.0754166756 - 0.1004338049j],
                             [0.1462524469 + 0.0706882969j], [-0.1044379785 - 0.055284761j],
                             [0.0723802168 + 0.0274865278j], [0.0797605965 - 0.1413665371j],
                             [-0.0214184203 - 0.048291773j], [-0.1145791866 + 0.0809761411j],
                             [-0.0647370616 - 0.0059933032j], [-0.0228904982 - 0.0908217543j]])

    print("Basis.py: there is no E1 designed for such parameters.")
    return None
