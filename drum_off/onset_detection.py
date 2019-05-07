'''
This module handles Onset Detection tasks
'''

from scipy.ndimage.filters import maximum_filter
from scipy.signal import argrelmax
import numpy as np

# superlux from madmom (Boeck et al)
def superflux(spec_x=[], A=None, B=None, win_size=8):
    """
    Calculate the superflux envelope according to Boeck et al.

    :param spec_x: optional, numpy array, A Spectrogram the superflux envelope is calculated from, X
    :param A: optional, numpy array, frequency response of the decomposed spectrogram, W
    :param B: optional, numpy array, activations of a decomposed spectrogram, H
    :param win_size: int, Hann window size, used to smooth the recomposition of

    :return: Superflux ODF of a spectrogram
    :notes: Must check inputs so that A and B have to be present together
    """

    # if A and B, the spec_x is recalculated
    if A is not None:
        # window function
        kernel = np.hamming(win_size)

        # apply window
        B = np.convolve(B, kernel, 'same')

        # rebuild spectrogram

        spec_x = np.outer(A, B)

    # To log magnitude
    spec_x = np.log(spec_x * 1 + 1)

    diff = np.zeros_like(spec_x.T)

    # Apply max filter
    max_spec = maximum_filter(spec_x.T, size=(1, 3))

    # Spectral difference
    diff[1:] = (spec_x.T[1:] - max_spec[: -1])

    # Keep only positive difference
    pos_diff = np.maximum(0, diff)

    # Sum bins
    sf = np.sum(pos_diff, axis=1)

    # normalize
    sf = sf / max(sf)

    # return ODF
    return sf


def pick_onsets(F, threshold=0.15, w=3.5):
    """
    Simple onset peak picking algorithm, picks local maxima that are
    greater than median of local maxima + correction factor.
    :param F: numpy array, Detection Function
    :param threshold: float, threshold correction factor
    :return: numpy array, peak indices
    """
    # Indices of local maxima in F
    localMaximaInd = argrelmax(F, order=1)

    # Values of local maxima in F
    localMaxima = F[localMaximaInd[0]]

    # Pick local maxima greater than threshold
    # (-.2 to move optimal threshold range away from zero in automatic threshold
    # calculation, This should not make a difference but it does, investigate)
    onsets = np.where(localMaxima >= threshold - .2)
    # Onset indices array
    rets = localMaximaInd[0][onsets]

    # Check that onsets are valid onsets
    i = 0
    while i in range(len(rets) - 1):
        # Check that the ODF goes under the threshold between onsets
        if F[rets[i]:rets[i + 1]].min() >= threshold:
            rets = np.delete(rets, i + 1)
        # Check that two onsets are not too close to each other
        elif rets[i] - rets[i + 1] > -w:
            rets = np.delete(rets, i + 1)
        else:
            i += 1
    # Return onset indices
    return rets


def pick_onsets_simpleDyn(F, threshold=0.15, N=2, w=3.5):
    """
    Simple onset peak picking algorithm, picks local maxima that are
    greater than median of local maxima + correction factor.
    :param F: numpy array, Detection Function
    :param threshold: float, threshold correction factor
    :return: numpy array, peak indices
    """
    # Indices of local maxima in F
    localMaximaInd = argrelmax(F, order=1)

    # Values of local maxima in F
    localMaxima = F[localMaximaInd[0]]

    # Simple dynamic threshold
    threshold_dyn = np.full((F.shape), threshold)
    for i in range(N, F.shape[0]):
        threshold_dyn[i] = threshold - .2 + 0.5 * np.mean(F[i - N:i])
    # Pick local maxima greater than threshold
    onsets = localMaxima >= threshold_dyn[localMaximaInd[0]]
    # Onset indices array
    rets = localMaximaInd[0][onsets]
    # print(threshold_dyn)
    # showEnvelope([F,threshold_dyn])
    # Check that onsets are valid onsets
    i = 0
    while i in range(len(rets) - 1):
        # Check that the ODF goes under the threshold between onsets
        if F[rets[i]:rets[i + 1]].min() >= threshold:
            rets = np.delete(rets, i + 1)
        # Check that two onsets are not too close to each other
        elif rets[i] - rets[i + 1] > -w:
            rets = np.delete(rets, i + 1)
        else:
            i += 1
    # Return onset indices
    return rets


def pick_onsets_dynT(F, threshold=0.15, N=10, w=3.5):
    """
    Peak picking with a dynamic threshold
    :param F: ODF
    :param delta: float, constant modifier for threshold
    :param N: int, number of frames the mean and median are calculated from
    :return: numpy array, peack indices
    """
    # Indices of local maxima in F
    localMaximaInd = argrelmax(F, order=1)

    # Values of local maxima in F
    localMaxima = F[localMaximaInd[0]]

    # Pick local maxima greater than threshold
    threshold_dyn = np.full((F.shape), threshold)
    #mova=movingAverage(F,N)
    for i in range(N, F.shape[0]):
        threshold_dyn[i] = threshold-0.2 + .25 * np.median(F[i - N:i]) + .25 * np.mean(F[i - N:i])
        #threshold_dyn[i] = threshold - 0.2 + .5 * mova[i]
    # showEnvelope(threshold)
    # showEnvelope(F)
    # threshold=thresh*np.median(localMaxima)+delta
    onsets = []
    onsets = localMaxima >= threshold_dyn[localMaximaInd[0]]
    # showEnvelope([F, threshold_dyn])
    rets = localMaximaInd[0][onsets]
    # remove peak if detFunc has not been under threshold.
    i = 0
    while i in range(len(rets) - 1):
        # Check that the ODF goes under the threshold between onsets
        if F[rets[i]:rets[i + 1]].min() >= threshold:
            rets = np.delete(rets, i + 1)
        # Check that two onsets are not too close to each other
        elif rets[i] - rets[i + 1] > -w:
            rets = np.delete(rets, i + 1)
        else:
            i += 1
    # Return onset indices
    return rets


def pick_onsets_bat(F, threshold=0.15, N=100, w=3.5, print=False):
    """
    Simple onset peak picking algorithm, picks local maxima that are
    greater than median of local maxima + correction factor.
    :param F: numpy array, Detection Function
    :param threshold: float, threshold correction factor
    :return: numpy array, peak indices
    """
    # Indices of local maxima in F
    localMaximaInd = argrelmax(F, order=1)

    # Values of local maxima in F
    localMaxima = F[localMaximaInd[0]]
    lamb = 1
    T1, T2, Tdyn = np.zeros((F.shape)), np.zeros((F.shape)), np.zeros((F.shape))
    # calculate every threshold
    # n = movingAverage(F, window=23)
    for i in range(N, len(F)):
        # Slow, find alternative if used
        percs = np.percentile(F[i - N:i], [25, 50, 75, 100])
        T1[i] = 0.5 * threshold + lamb * (percs[2] - percs[0]) + percs[1]
        T2[i] = threshold * percs[3]
        p = 2
        # final Dyn threshold from battenberg
        Tdyn[i] = ((T1[i] ** p + T2[i] ** p) / 2.) ** (1. / p)

    onsets = localMaxima >= Tdyn[localMaximaInd[0]]
    # Onset indices array
    rets = localMaximaInd[0][onsets]
    # Check that onsets are valid onsets
    i = 0
    while i in range(len(rets) - 1):
        # Check that the ODF goes under the threshold between onsets
        if F[rets[i]:rets[i + 1]].min() >= threshold:
            rets = np.delete(rets, i + 1)
        # Check that two onsets are not too close to each other
        elif rets[i] - rets[i + 1] > -w:
            rets = np.delete(rets, i + 1)
        else:
            i += 1
    # Return onset indices
    return rets

def energyDifference(signal, win_size=8):
    """
    Energy difference ODF for time domain signals such as the activations from NMFD

    :param signal: numpy array, H from NMFD
    :param win_size: int, size of the smoothing window

    :return: numpy array, ODF
    """
    # window
    kernel = np.hamming(win_size)

    # apply window
    signal = np.convolve(signal, kernel, 'same')

    # calculate Energy difference
    signal[1:] = signal[1:] - signal[:-1]

    # Half wave rectify
    signal = (signal + np.abs(signal)) / 2

    # normalize
    signal = signal / max(signal)

    # return ODF
    return signal