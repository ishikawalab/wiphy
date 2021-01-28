# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

import os

import numpy as np
from scipy.interpolate import interp1d
from matplotlib.patches import ConnectionPatch

__all__ = ['rcParamsAcademic', 'getXCorrespondingToY', 'getYCorrespondingToX',
           'trimber', 'plotber', 'addText', 'addArrow', 'bib']


def rcParamsAcademic(plt):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.color'] = 'black'
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['markers.fillstyle'] = 'none'


def getXCorrespondingToY(xarr, yarr, targety):
    if targety < np.min(yarr) or targety > np.max(yarr):
        return np.NaN
    spfunc = interp1d(yarr, xarr)
    return spfunc(targety)


def getYCorrespondingToX(xarr, yarr, targetx):
    if targetx < np.min(xarr) or targetx > np.max(xarr):
        return np.NaN
    spfunc = interp1d(xarr, yarr)
    return spfunc(targetx)


def trimber(ax, x, y):
    threshold = ax.get_ylim()[0]
    inds = np.where(y >= threshold)[0]
    return x[inds], y[inds]


def plotber(ax, d, **kwargs):
    if isinstance(d, dict):
        x, y = d["snr_dB"], d["ber"]
    if isinstance(d, np.ndarray):
        x, y = d[:, 0], d[:, 1]

    snrfrom = ax.get_xlim()[0]
    snrto = ax.get_xlim()[1]
    if np.min(x) < snrfrom:
       inds = np.where(x >= snrfrom)
       x = x[inds]
       y = y[inds]

    if np.max(x) > snrto:
       inds = np.where(x <= snrto)
       x = x[inds]
       y = y[inds]

    ms = 6.0 * kwargs["ms"] if "ms" in kwargs else 6 # default = 6
    # line
    ax.plot(x, y, clip_on=True, color=kwargs["color"], marker="",
            linestyle=kwargs["linestyle"])
    x, y = trimber(ax, x, y)
    # marker
    ax.plot(x, y, clip_on=False, color=kwargs["color"], marker=kwargs["marker"], ms = ms, linestyle="")
    # label
    if "label" in kwargs:
       ax.plot([], [], color=kwargs["color"], marker=kwargs["marker"], ms = ms,
                   linestyle=kwargs["linestyle"], label=kwargs["label"])


def addText(ax, cx, cy, text, color="k", ha = "left", va = "center"):
    ax.annotate(text, xy=(cx, cy), ha=ha, va=va, color=color)


def addArrow(ax, cx, xdiff, cy, ydiff, text, color="k", ha = "left", va = "center",
             arrowstyle = "-|>", shrinkA = 2, shrinkB = 4, xlog = False, ylog = True):
    x = cx * xdiff if xlog else cx + xdiff
    y = cy * ydiff if ylog else cy + ydiff
    con = ConnectionPatch(
        xyA = (x, y), xyB = (cx, cy), coordsA="data", coordsB="data",
        arrowstyle=arrowstyle,shrinkA=shrinkA, shrinkB=shrinkB, mutation_scale=20, fc="w", color = color)
    # connectionstyle="arc3,rad=0.3"
    ax.add_artist(con)
    ax.annotate(text, xy=(x, y), ha=ha, va=va, color=color)#,box=dict(boxstyle="round", fc="w")
    #ax.plot([cx], [cy], color=color, marker="o", markersize=8, clip_on=False, linestyle="")


def bib(APATH, label):
    import re
    if not os.path.exists(APATH):
        # print("The given APATH does not exist: APATH = " + APATH)
        return label

    chit = re.search(r'\\cite{(\S+)}', label)
    if chit == None:
        # print("The given label does not match: label = " + label)
        return label

    bibkey = chit.group(1)

    with open(APATH, mode='r') as f:
        auxstr = f.read()
        ahit = re.search('bibcite{' + bibkey + '}{(\d+)}', auxstr)
        if ahit:
            cn = ahit.group(1)
        else:
            ahit = re.search('bibcite{' + bibkey + '}{{(\d+)}', auxstr)
            if ahit:
                cn = ahit.group(1)
            else:
                cn = "?"

        return label.replace(chit.group(0), "[" + cn + "]")
