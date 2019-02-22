'''
This module handles the Tempo Tracking, Beat Tracking and Quantization functionality
'''
from constants import *
from scipy import fftpack as fft
import numpy as np
from utils import frame_to_time, time_to_frame

def extract_tempo(onsets=None, win_len_s=3, smooth_win_scalar=2, constant_tempo=True, h=HOP_SIZE):
    """
    Tempo extraction method modified version of the librosa library tempo extraction
    :param onsets: numpy array, novelty function
    :param window_size_in_s: int, tempo extraction window length
    :param constant_tempo: boolean, if True one tempo is used for all frames.
    :return: numpy array, a list of tempi
    """
    print('called')
    #Window in samples
    fps=SAMPLE_RATE/h
    N = int(fps * win_len_s)
    #max_bpm = int(DEFAULT_TEMPO * 4)
    max_bpm=N
    pad_len = int(N / 2 + 1)  # fixed pad length

    onsets = np.pad(onsets, pad_len,
                    mode='mean')
    #Stride_tricks wiev of onsets, serious speedup.
    n_frames = 1 + int((len(onsets) - N))
    fonsets = np.lib.stride_tricks.as_strided(onsets, shape=(N, n_frames),
                                              strides=(onsets.itemsize, onsets.itemsize))

    #dft, only processing the tempo range.
    fftonsets = fft.fft(fonsets, n=max_bpm, axis=0,overwrite_x=True)

    #dft times it's complex conjugate
    comp = fftonsets * np.conj(fftonsets)

    #inverse of comp
    autocorr = fft.ifft(comp, axis=0, overwrite_x=True)

    #normalization and remove the complex values
    sff = (autocorr.real / autocorr.real.max())
    #mean value
    sff_mean = np.mean(sff, axis=1, keepdims=True)

    #frequency to bpm bins
    #This divides by zero :D Well i never...
    bpms = 60 * fps / np.array(range(1,max_bpm))
    bpms=np.concatenate((np.array([0.0000001]),bpms))
    #gaussian weighing
    prior_mean = np.exp(-0.5 * ((np.log2(bpms) - np.log2(DEFAULT_TEMPO)) / 1.) ** 2)

    #Mean tempo
    bpm_mean = np.argmax(sff_mean * prior_mean[:, np.newaxis], axis=0)

    # first find the most common tempo of the take
    tempi_mean = bpms[bpm_mean]
    # then force the tempi to that area with the weighing function
    prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(tempi_mean)) / .2) ** 2)
    loc_max = np.argmax(sff * prior[:, np.newaxis], axis=0)
    tempi = bpms[loc_max]
    #If ran twice gaps occur in tempotrack, set gaps to 120
    tempi[loc_max == 0] = DEFAULT_TEMPO
    # Get constant tempo estimate
    if constant_tempo:
        tempi[:] = tempi_mean
        tempi_smooth = tempi

    #or smooth the local tempi
    else:
        #6s. window
        kernel = np.hanning(int(N*smooth_win_scalar))

        #Pad edges
        tempi_pad = np.pad(tempi, pad_len, mode='mean')

        #Smooth and remove padding to align tempi to original beat
        tempi_smooth = np.convolve(tempi_pad / kernel.sum(), kernel, 'same')
        tempi_smooth=np.roll(tempi_smooth,-pad_len)

    #Constant target, could scale to manifolds here also... targets=[default/2, default, default*2,...]
    targets = [DEFAULT_TEMPO]
    target_tempos = []
    # Change tempi to tempo quantization multipliers
    for i in range(tempi_smooth.size):
        target_tempos.append(min(targets, key=lambda x: abs(x - tempi_smooth[i])))

    #median for the mean tempo transform
    target_median = np.median(target_tempos)

    #transform the smoothed tempi to scale factors from DEFAULT_TEMPO for quantization
    tempi_smooth[:] = tempi_smooth[:] / target_median

    return (tempi_smooth)

def conform_time(X, tempomap, h=HOP_SIZE):
    """
    Conforms the hits X according to a tempomap
    :param X: numpy array, The onsets to quantize
    :param tempomap: numpy array, the tempo modifiers for each frame
    :param quantize: boolean, quantize the conformed X to grid of 16th notes at 4 times the default bpm
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

    newgap = frame_to_time(sum(tempomap[:X[0]]),hop_length=h)
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
    frame_retX = time_to_frame(retX, hop_length=Q_HOP).astype(int)
    return frame_retX

#Ellis dp pseudocode to python
def beatSimpleDP(onsets, alpha=100):
    period=round(60.0 * SAMPLE_RATE/Q_HOP/DEFAULT_TEMPO)
    backlink = np.full(onsets.size, -1)
    cumscore = onsets
    prange = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int)
    txcost= (-alpha*abs((np.log(prange/-period))**2))
    for i in range(max(-prange + 1),onsets.size):
        timerange = i + prange
        scorecands = txcost + cumscore[timerange]
        vv,xx = max(scorecands),scorecands.argmax()
        cumscore[i] = vv + onsets[i]
        backlink[i] = timerange[xx]
    beats = [cumscore.argmax()]
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])
    beats = np.array(beats[::-1], dtype=int)
    return beats

def two_fold_quantize(onsets, drumkit):
    onsets = onsets / onsets.max()
    deltaTempo = extract_tempo(onsets, constant_tempo=False)
    newmax = 0
    for i in drumkit:
        hits = i.get_hits()
        if len(hits) is None or len(hits) == 0:
            i.set_hits([])
        else:
            i.set_hits(conform_time(hits, deltaTempo))
            if newmax < max(i.get_hits()):
                newmax = max(i.get_hits())

    if True:
        beats = np.zeros(newmax + 1)
        for i in drumkit:
            # create an odf from already found onsets weigh toward kick and snare
            beats[i.get_hits()] += 1 - (i.get_name()[0] / (len(drumkit)))
        trimmed_beats = np.trim_zeros(beats)
        tracked_beats = beatSimpleDP(trimmed_beats, alpha=100)
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
