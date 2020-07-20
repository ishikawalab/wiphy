# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import os

import numpy as np
from scipy.interpolate import interp1d

__all__ = ['getXCorrespondingToY', 'bib']


def getXCorrespondingToY(xarr, yarr, y):
    if y < np.min(yarr) or y > np.max(yarr):
        return np.NaN
    spfunc = interp1d(yarr, xarr)
    return spfunc(y)


def bib(APATH, label):
    import re
    if not os.path.exists(APATH):
        return label

    with open(APATH, mode='r') as f:
        #
        chit = re.search(r'\\cite{(\S+)}', label)
        if chit == None:
            return label
        bibkey = chit.group(1)

        #
        ahit = re.search(bibkey + '}{(\d+)}', f.read())
        if ahit:
            cn = ahit.group(1)
        else:
            cn = "?"

        return label.replace(chit.group(0), "[" + cn + "]")
