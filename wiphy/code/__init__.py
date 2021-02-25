# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

__all__ = ['modulator', 'ostbc', 'duc', 'im', 'adsm', 'anm', 'tast', 'basis', 'generateCodes']

from . import modulator
from . import adsm
from . import anm
from . import basis
from . import duc
from . import im
from . import ostbc
from . import tast


def generateCodes(params):
    """
    Generates a codebook based on the given dict.

    Args:
        params (dict): a dict that contains parameters.

    Returns:
        ndarray: the generated codebook.

    """
    scode = params["code"].lower()

    if scode == "symbol":
        return modulator.generateSymbolCodes(params["mod"], params["L"])
    elif scode == "blast":
        return im.generateIMCodes("dic", params["M"], params["M"], 1, params["mod"], params["L"], meanPower=1.0)
    elif scode == "index":
        return im.generateIMCodes(params["dm"], params["M"], params["K"], params["Q"], params["mod"], params["L"],
                                  meanPower=1.0)
    elif scode == "ostbc":
        if "O" in params:
            return ostbc.generateOSTBCodes(params["M"], params["mod"], params["L"], params["O"])
        else:
            return ostbc.generateOSTBCodes(params["M"], params["mod"], params["L"])
    elif scode == "duc":
        return duc.generateDUCCodes(params["M"], params["L"])
    elif scode == "adsm":
        if "u1" in params:
            return adsm.generateADSMCodes(params["M"], params["mod"], params["L"], params["u1"])
        else:
            return adsm.generateADSMCodes(params["M"], params["mod"], params["L"])
    elif scode == "tast":
        return tast.generateTASTCodes(params["M"], params["Q"], params["L"], params["mod"])
    elif scode == "anm":
        return anm.generateANMCodes(params["M"], params["mod"], params["L"])
