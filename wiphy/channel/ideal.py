# Copyright (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt

__all__ = ['generateAWGNChannel', 'generateRayleighChannel', 'generateRayleighOFDMChannel',
           'getPositionsUniformLinearArray', 'getPositionsRectangular2d', 'generateRicianLoSChannel',
           'generateRicianChannel']

import os

from ..util.general import randn_c

import numpy as np


def generateAWGNChannel(N, M, IT=1):
    """
    Generates the identical matrix to simulate the additive white Gaussian noise (AWGN) channel.
    Args:
        N (int): dummy variable.
        M (int): the number of transmit streams.
        IT (int): the number of parallel channel matrices.

    Returns:
        ndarray: a (IT*M)xM vertical identical matrix.
    """
    return np.tile(np.eye(M), IT).T


def generateRayleighChannel(N, M, IT=1):
    """
    Generates the ideal Rayleigh fading channel coefficients.

    Args:
        N (int):the number of receive antennas.
        M (int): the number of transmit antennas.
        IT (int): the number of parallel channel matrices.

    Returns:
        ndarray: a (IT*N)xM Rayleigh channel matrix.
    """
    return randn_c(IT * N, M)


def generateRayleighOFDMChannel(M, IT=1):
    """
    Generates the ideal Rayleigh fading channel coefficients for OFDM scenarios. The channel matrix becomes diagonal.

    Args:
        M (int): the number of subcarriers.
        IT (int): the number of parallel channel matrices.

    Returns:
        ndarray: a (IT*M)xM channel matrix.
    """
    return (np.tile(np.eye(M), IT) * randn_c(IT * M)).T


def getPositionsUniformLinearArray(Nae, ae_spacing, height):
    x = np.arange(Nae) * ae_spacing
    x -= np.mean(x)  # centering
    y = np.zeros(Nae)
    z = np.repeat(np.array(height), Nae)

    return x, y, z


def getPositionsRectangular2d(Nae, ae_spacing, height):
    sq = np.floor(np.sqrt(Nae))
    x = np.arange(Nae) % sq
    y = np.floor(np.arange(Nae) / sq)
    z = np.repeat(np.array(height), Nae)

    x *= ae_spacing
    y *= ae_spacing

    x -= np.mean(x)
    y -= np.mean(y)

    return x, y, z


def generateRicianLoSChannel(tx, ty, tz, rx, ry, rz, wavelength, IT=1):
    """
    Args:
        tx (numpy.array): the x positions of transmit antenna elements.
        ty (numpy.array): the y positions of transmit antenna elements.
        tz (numpy.array): the z positions of transmit antenna elements.
        rx (numpy.array): the x positions of receive antenna elements.
        ry (numpy.array): the y positions of receive antenna elements.
        rz (numpy.array): the z positions of receive antenna elements.
        wavelength (float): the wavelength.
        IT (int): the number of parallel channel matrices.
    """

    M = len(tx)  # the number of transmit antenna elements
    N = len(rx)  # the number of receive antenna elements

    r = np.zeros((N, M), dtype=complex)
    for n in range(N):
        for m in range(M):
            r[n][m] = np.sqrt(np.square(rx[n] - tx[m]) + np.square(ry[n] - ty[m]) + np.square(rz[n] - tz[m]))

    anHLoS = np.exp(-1j * 2.0 * np.pi / wavelength * r)
    return np.tile(anHLoS.T, IT).T  # IT \cdot N \times M


def generateRicianChannel(HLoS, N, M, K_dB, IT=1):
    """
    Generates the ideal Rician fading channel coefficients.

    Args:
        HLoS (numpy.ndarray): the LoS components.
        N (int):the number of receive antennas.
        M (int): the number of transmit antennas.
        K_dB (float): the rician K factor in dB.
        IT (int): the number of parallel channel matrices.

    Returns:
        ndarray: a (IT*N)xM Rayleigh channel matrix.
    """
    K = 10 ** (K_dB / 10.0)
    return np.sqrt(K / (1.0 + K)) * HLoS + randn_c(IT * N, M) / np.sqrt(1.0 + K)
