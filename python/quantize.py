'''
This module handles the Tempo Tracking, Beat Tracking and Quantization functionality
'''
import numpy as np
from scipy import fftpack as fft

from constants import *
from utils import frame_to_time, time_to_frame


def extract_tempo(onsets=None, win_len_s=3, smooth_win_scalar=2, constant_tempo=True, h=HOP_SIZE):
    """
    Tempo extraction method modified version of the librosa library tempo extraction
    :param onsets: numpy array, novelty function
    :param win_len_s: int, tempo extraction window length
    :param smooth_win_scalar: int, multiplier of win_len_s for the tempi smoothing window
    :param constant_tempo: boolean, if True mean tempo is used for all frames.
    :param h: int, hop length
    :return: numpy array, a list of tempi
    """
    # Window in samples
    fps = SAMPLE_RATE / h
    N = int(fps * win_len_s)
    # Get autocorrelation tempogram
    autocorr = get_fft(onsets, N, fps, return_autoc=True)

    # normalization and remove the complex values
    sff = np.abs(autocorr)

    # PLP Test, not worth it, 0 improvement
    # sff= extract_plp(onsets, win_len_s=win_len_s, h=HOP_SIZE)

    # mean value
    sff_mean = np.mean(sff, axis=1, keepdims=True)

    # frequency to bpm bins
    bpms = 60 * fps / np.array(range(1, N))
    bpms = np.concatenate((np.array([0.0000001]), bpms))
    # gaussian weighing
    prior_mean = np.exp(-0.5 * ((np.log2(bpms) - np.log2(DEFAULT_TEMPO)) / 1.) ** 2)

    # Mean tempo
    bpm_mean = np.argmax(sff_mean * prior_mean[:, np.newaxis], axis=0)

    # first find the most common tempo of the take
    tempi_mean = bpms[bpm_mean]
    # then force the tempi to that area with the weighing function
    prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(tempi_mean)) / .2) ** 2)
    loc_max = np.argmax(sff * prior[:, np.newaxis], axis=0)
    tempi = bpms[loc_max]
    # If ran twice gaps occur in tempotrack, set gaps to 120
    tempi[loc_max == 0] = DEFAULT_TEMPO
    # Get constant tempo estimate
    if constant_tempo:
        tempi[:] = tempi_mean
        tempi_smooth = tempi

    # or smooth the local tempi
    else:
        # 6s. window
        kernel = np.hanning(int(N * smooth_win_scalar))

        # Pad edges
        tempi_pad = np.pad(tempi, kernel.size, mode='mean')
        # Smooth and remove padding to align tempi to original beat
        tempi_smooth = np.convolve(tempi_pad / kernel.sum(), kernel, 'same')
        tempi_smooth = np.roll(tempi_smooth, -kernel.size)
        tempi_smooth=tempi_smooth[:tempi.size]
    # Constant target, could scale to manifolds here also... targets=[default/2, default, default*2,...]
    targets = [DEFAULT_TEMPO]
    target_tempos = []
    # Change tempi to tempo quantization multipliers
    for i in range(tempi_smooth.size):
        target_tempos.append(min(targets, key=lambda x: abs(x - tempi_smooth[i])))

    # median for the mean tempo transform
    target_median = np.median(target_tempos)

    # transform the smoothed tempi to scale factors from DEFAULT_TEMPO for quantization
    tempi_smooth[:] = tempi_smooth[:] / target_median

    return (tempi_smooth)


def get_fft(signal, N=100, fps=100, return_autoc=False, return_half_spectrum=False):
    """
    Calculates a tempogram from a novelty function,
    :param signal: numpy array, Novelty function
    :param N: int, Window size in frames
    :param fps: float, frames per second
    :param return_autoc: Boolean, return Autocorrelation Tempogram
    :param return_half_spectrum: Boolean, return bottom half of spectrum only
    :return: numpy complex array, Tempogram
    """
    ##nr. frequency bins = Half of FRAME_SIZE

    fs = N
    # hop size
    hs = 1
    # frequencies
    n_freqs = fs
    n_ret = n_freqs
    if return_half_spectrum:
        n_ret = int(fs / 2)
    # HOP_LENGTH, no 1 frame, spaced index
    frames_index = np.arange(0, len(signal), hs)
    # +2 frames to correct NMF systematic errors...
    err_corr = 0
    data = np.zeros((len(frames_index) + err_corr, n_freqs), dtype=np.complex64)
    # Window
    win = np.hamming(fs)
    # STFT
    for frame in range(len(frames_index)):
        # Get one frame length audio clip
        one_frame = signal[frames_index[frame]:frames_index[frame] + fs]
        # Pad last frame if needed
        if one_frame.shape[0] < fs:
            one_frame = np.pad(one_frame, (0, fs - one_frame.shape[0]), 'mean')
        # apply window
        fft_frame = np.multiply(one_frame, win)
        # FFT
        data[frame + err_corr] = np.fft.fft(fft_frame, fs, axis=0)
    if return_autoc:
        comp = np.abs(data) ** 2
        autoc = fft.ifft(comp, axis=-1)
        if return_half_spectrum:
            return autoc[:, :n_ret].T
        else:
            return autoc.T
    if return_half_spectrum:
        return data[:, n_ret:].T
    else:
        return data.T



def conform_time(X, tempomap, h=HOP_SIZE, preserve_resolution=False):
    """
    Conforms the hits X according to a tempomap
    :param X: numpy array, The onsets to quantize
    :param tempomap: numpy array, the tempo modifiers for each frame
    :return: numpy array, quantized onsets

    Notes:
    """
    # Shortest allowed note in seconds 16th note at 480bpm The high resolution allows more expression.
    # Less resolution would result in poor performances if the player did not play in exellent time.
    if not len(X):
        return []
    shortest_note = DEFAULT_TEMPO / 60 / SXTH_DIV / 4
    # return value space
    retX = np.zeros((X.size))
    # gap of beginning to the first hit as time value
    X = X.astype(int)

    newgap = frame_to_time(sum(tempomap[:X[0]]), hop_length=h)
    # newgap = np.rint(newgap / shortest_note) * shortest_note
    # store first hit
    retX[0] = newgap
    # retX[0] = np.rint(retX[0] / shortest_note) * shortest_note
    # iterate over all hits

    for i in range(1, X.size):
        # Calculate the gap between two consecutive hits
        newgap = frame_to_time(sum(tempomap[X[i - 1]:X[i]]), hop_length=h)
        # newgap = np.rint(newgap / shortest_note) * shortest_note
        # move the hit to last hit+newgap
        retX[i] = retX[i - 1] + newgap
    # if hits are to be quantized
    if preserve_resolution:
        frame_retX = time_to_frame(retX, hop_length=h).astype(int)
    else:
        frame_retX = time_to_frame(retX, hop_length=Q_HOP).astype(int)
    return frame_retX


# Ellis dp pseudocode to python
def beatSimpleDP(onsets, alpha=680):
    """
    Beat detection by dynamic programming
    :param onsets: numpy array, Novelty function
    :param alpha: Int, transition weight, 680 from original paper.
    :return: numpy array, detected beat locations.
    """
    period = round(60.0 * SAMPLE_RATE / Q_HOP / DEFAULT_TEMPO)
    backlink = np.full(onsets.size, -1)
    cumscore = onsets.copy() #copy if we want to play with novelty function afterwards
    prange = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int) #if we had tempi curve slices here could we skip step 1?
    txcost = (-alpha * abs((np.log(prange / -period)) ** 2))
    for i in range(max(-prange + 1), onsets.size):
        timerange = i + prange
        scorecands = txcost + cumscore[timerange]
        vv, xx = max(scorecands), scorecands.argmax()
        cumscore[i] = vv + onsets[i]
        backlink[i] = timerange[xx]
    beats = [cumscore.argmax()]
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])
    beats = np.array(beats[::-1], dtype=int)
    return beats


def two_fold_quantize(onsets, drumkit, quant_factor):
    # Normalize
    onsets = onsets / onsets.max()
    # remove noise floor
    onsets = onsets - .1
    # clip
    np.clip(onsets, 0, 1, onsets)

    # test the method
    # extract_plp(onsets)

    # Get tempomap
    deltaTempo = extract_tempo(onsets, constant_tempo=False)

    # Smooth out tempo fluctuation and quantize around 120 bpm
    if quant_factor > 0.:
        newmax = 0
        for i in drumkit:
            hits = i.get_hits()
            if len(hits) is None or len(hits) == 0:
                i.set_hits([])
            else:
                i.set_hits(conform_time(hits, deltaTempo))
                if newmax < i.get_hits()[-1]:
                    newmax = i.get_hits()[-1]

    # Find beat positions and move to 120bpm grid
    if quant_factor > .5:
        newmax = 0
        for i in drumkit:
            try:
                newmax = max(newmax,i.get_hits()[-1]) #was max(i.get_hits()) chagend to [-1].
            except:  # if a drum is not present in the take
                newmax = newmax
        beats = np.zeros(newmax + 1)
        for i in drumkit:
            # create an odf from already found onsets weigh toward kick and snare
            beats[i.get_hits()] += 1 - (i.get_name()[0] / (len(drumkit)))
        trimmed_beats = np.trim_zeros(beats)

        tracked_beats = beatSimpleDP(trimmed_beats, alpha=680)
        beat_interval = round((tracked_beats.max() - tracked_beats.min()) / tracked_beats.size)
        fixed_beats = range(tracked_beats.min(), tracked_beats.max() + 100, int(beat_interval))
        beat_diff = np.zeros(tracked_beats.size)
        for i in range(tracked_beats.size):
            beat_diff[i] = fixed_beats[i] - tracked_beats[i]
        for i in drumkit:
            quant_hits = []
            for j in i.get_hits():
                a = (np.abs(tracked_beats - j)).argmin()
                quant_hits.append(j + int(beat_diff[a]))
            i.set_hits(quant_hits)
    return drumkit, deltaTempo

# # stab in the PLP way, unfinished business...
# #import matplotlib.cm as cmaps
# #import matplotlib.pyplot as plt
# def extract_plp(onsets=None, win_len_s=6, h=HOP_SIZE):
#     """
#     Tempo extraction method modified version of the librosa library tempo extraction
#     :param onsets: numpy array, novelty function
#     :param window_size_in_s: int, tempo extraction window length
#     :param constant_tempo: boolean, if True one tempo is used for all frames.
#     :return: numpy array, a list of tempi
#     """
#
#     fps = SAMPLE_RATE / h
#     N = int(fps * win_len_s)
#     bpms = 60 * fps / np.array(np.arange(1, N))
#     bpms = np.concatenate((np.array([0]), bpms))
#     fftbpms = np.exp(bpms[:])
#     bpms = bpms[:258]
#
#     # plt.figure(figsize=(10,3))
#     # plt.plot(onsets[200:1200])
#     # plt.show()
#     #print(fftbpms)
#     fftonsets = get_fft(onsets, N, fps, return_half_spectrum=True)
#     #plt.figure(figsize=(10,3))
#     #plt.imshow(np.abs(fftonsets), aspect='auto', origin='lower',cmap=cmaps.get_cmap('inferno'))
#     #plt.yticks(np.arange(0,250,10)[::1],fftbpms.astype(int))
# #
#     #plt.show()
#     fftonsets = get_fft(onsets, N, fps, return_autoc=True, return_half_spectrum=True)
#     #plt.figure(figsize=(10, 3))
#     #plt.imshow(np.flipud(np.abs(fftonsets)), aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
#     #plt.yticks(np.arange(0,250,5),bpms[::-5].astype(int))
#     #plt.show()
#     sff = np.abs(fftonsets)
#     sff_mean = np.mean(sff, axis=1, keepdims=True)
#     # frequency to bpm bins
#
#     # gaussian weighing
#     prior_mean = np.exp(-0.5 * ((np.log2(bpms) - np.log2(DEFAULT_TEMPO)) / 1.) ** 2)
#     # Mean tempo
#     bpm_mean = np.argmax(sff_mean * prior_mean[:, np.newaxis], axis=0)
#     # first find the most common tempo of the take
#     tempi_mean = bpms[bpm_mean]
#     print(tempi_mean, bpm_mean)
#     # then force the tempi to that area with the weighing function
#     prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(tempi_mean)) / .2) ** 2)
#     loc_max = np.argmax(sff * prior[:, np.newaxis], axis=0)
#     # loc_max = np.argmax(fftonsets)
#     tempi = bpms[loc_max]
#     # plt.figure(figsize=(10,2))
#     # plt.plot(tempi)
#     # plt.show()
#     phases = np.zeros_like(tempi)
#     for i in range(loc_max.shape[0]):
#         phases[i] = (1 / (2 * np.pi)) * np.arccos(fftonsets[loc_max[i], i].real / np.abs(fftonsets[loc_max[i], i]))
#
#     win_len = N
#     W = np.hanning(win_len)
#
#     plp = np.zeros_like(tempi)
#     for i in range(N, tempi.shape[0] - N):
#         t0 = int(np.ceil(i - win_len / 2))
#         t1 = int(np.floor(i + win_len / 2))
#         cosine = W * np.cos(2 * np.pi * (tempi[t0:t1] - phases[t0:t1]))
#         np.clip(cosine, 0, 1, cosine)
#         plp[t0:t1] += cosine
#
#     fftplp = get_fft(plp, N, fps, return_autoc=True)
#     sff = np.abs(fftplp)
#     return sff
#
#
# #

