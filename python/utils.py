import time

import librosa
import madmom
import numpy as np
import pandas as pd
import scipy
from scipy import fftpack as fft
from scipy.ndimage.filters import median_filter, maximum_filter
from scipy.signal import argrelmax,stft
from sklearn.cluster import KMeans

FRAME_SIZE = 2 ** 10
HOP_SIZE = 2 ** 9
SAMPLE_RATE = 44100
FREQUENCY_PRE = np.ones((24))  # [0,16384]#0-2^14
MIDINOTE = 36  # kickdrum in most systems
THRESHOLD = 0.0
PROBABILITY_THRESHOLD = 0.0
DEFAULT_TEMPO = 120
DRUMKIT_PATH = '../trainSamplet/'
REQUEST_RESULT = False
DELTA = 0.15
midinotes = [36, 38, 42, 46, 50, 43, 51, 49, 44]  # BD, SN, CHH, OHH, TT, FT, RD, CR, SHH, Here we need generality
nrOfDrums = 24  # Maximum kit size
nrOfPeaks = 32
ConvFrames = 10
K = 1
MS_IN_MIN = 60000
SXTH_DIV = 16
QUANTIZE = False
ENCODE_PAUSE = True
_ImRunning = False
proc = madmom.audio.filters.BarkFilterbank(
    madmom.audio.stft.fft_frequencies(num_fft_bins=int(FRAME_SIZE / 2), sample_rate=SAMPLE_RATE), num_bands='double')


class Drum(object):
    """
    A Drum is any user playable drumkit part representation

    Parameters
    ----------
    name : String
        Name of the drum
    frequency_pre : list, optional
        corner frequencies of drum signal
    midinote: int, optional
        midi note representing the drum
    threshold : float, optional
        onset detection threshold.
    probability_threshold : float, optional
        NN prediction threshold.


    Notes
    -----
    Unfinished class, work in progress

    """

    def __init__(self, name, highEmph, peaks, samples=None, templates=None, frequency_pre=FREQUENCY_PRE,
                 midinote=MIDINOTE, threshold=THRESHOLD, probability_threshold=PROBABILITY_THRESHOLD
                 , hitlist=None, **kwargs):

        # set attributes
        self.name = name
        self.highEmph = highEmph
        self.peaks = peaks
        if len(frequency_pre):
            self.frequency_pre = frequency_pre
        if len(samples):
            self.samples = samples
        if len(templates):
            self.templates = templates
        if midinote:
            self.midinote = midinote
        if threshold:
            self.threshold = float(threshold)
        else:
            self.threshold = 0.
        if probability_threshold:
            self.probability_threshold = float(probability_threshold)
        if hitlist:
            self.hitlist = hitlist

    def set_hits(self, hitlist):
        self.hitlist = hitlist

    def get_hits(self):
        return self.hitlist

    def concat_hits(self, hitlist):
        self.hitlist = np.concatenate((self.get_hits(), hitlist))

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_highEmph(self, highEmph):
        self.highEmph = highEmph

    def get_highEmph(self):
        return self.highEmph

    def set_peaks(self, peaks):
        self.name = peaks

    def get_peaks(self):
        return self.peaks

    def set_templates(self, templates):
        self.templates = templates

    def get_templates(self):
        return self.templates

    def set_samples(self, samples):
        self.samples = samples

    def get_samples(self):
        return self.samples

    def set_frequency_pre(self, frequency_pre):
        self.frequency_pre = frequency_pre

    def get_frequency_pre(self):
        return self.frequency_pre

    def set_midinote(self, midinote):
        self.midinote = int(midinote)

    def get_midinote(self):
        return self.midinote

    def set_threshold(self, threshold):
        self.threshold = float(threshold)

    def get_threshold(self):
        return self.threshold

    def set_probability_threshold(self, probability_threshold):
        self.probability_threshold = float(probability_threshold)

    def get_probability_threshold(self):
        return self.probability_threshold


def to_midinote(notes):
    return list(midinotes[i] for i in notes)


def findDefBins(frames, filteredSpec):
    """
    Calculate the prior vectors for W to use in NMF
    :param frames: Numpy array of hit locations (frame numbers)
    :param filteredSpec: Spectrogram, the spectrogram where the vectors are extracted from
    :return: tuple of Numpy arrays, prior vectors Wpre,heads for actual hits and tails for decay part of the sound
     """
    gaps = np.zeros((frames.shape[0], ConvFrames))
    for i in range(frames.shape[0]):
        for j in range(gaps.shape[1]):
            gaps[i, j] = frames[i] + j

    a = np.reshape(filteredSpec[gaps.astype(int)], (nrOfPeaks, -1))
    kmeans = KMeans(n_clusters=K).fit(a)

    heads = np.zeros((proc.shape[1], ConvFrames, K))
    for i in range(K):
        heads[:, :, i] = np.reshape(kmeans.cluster_centers_[i], (proc.shape[1], ConvFrames), order='F')

    tailgaps = np.zeros((frames.shape[0], ConvFrames))
    for i in range(frames.shape[0]):
        for j in range(gaps.shape[1]):
            tailgaps[i, j] = frames[i] + j + ConvFrames

    a = np.reshape(filteredSpec[tailgaps.astype(int)], (nrOfPeaks, -1))
    kmeans = KMeans(n_clusters=K).fit(a)

    tails = np.zeros((proc.shape[1], ConvFrames, K))
    for i in range(K):
        tails[:, :, i] = np.reshape(kmeans.cluster_centers_[i], (proc.shape[1], ConvFrames), order='F')

    return (heads, tails)


def getStompTemplate(numHits=2):
    """

    :param numHits:
    :return:
    """
    global _ImRunning

    _ImRunning = True
    stompResolution = 1
    buffer = np.zeros(shape=(2277920))
    j = 0
    time.sleep(0.1)
    strm = madmom.audio.signal.Stream(sample_rate=SAMPLE_RATE, num_channels=1, frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    for i in strm:
        # print(i.shape)
        buffer[j:j + HOP_SIZE] = i[:HOP_SIZE]
        j += HOP_SIZE
        if j >= 2217920 or (not _ImRunning):
            buffer[j:j + 6000] = np.zeros(6000)
            # buffer=madmom.audio.signal.normalize(buffer)
            peaks, bins, threshold = getPeaksFromBuffer(buffer[:j + 6000], stompResolution, numHits)
            strm.close()
            return peaks, bins, threshold, buffer[:j + 6000]


def getPeaksFromBuffer(buffer, resolution, numHits):
    filt_spec = get_preprocessed_spectrogram(buffer)
    threshold = 0.5
    searchSpeed = .1
    # peaks=cleanDoubleStrokes(madmom.features.onsets.peak_picking(superflux_3,threshold),resolution)
    H0 = superflux(spec_x=filt_spec.T)
    peaks = cleanDoubleStrokes(pick_onsets(H0, delta=threshold), resolution)
    changed = False
    while (peaks.shape != (numHits,)):
        # Make sure we don't go over numHits
        # There is a chance of an infinite loop here!!! Make sure that don't happen
        if (peaks.shape[0] > numHits):
            if changed == False:
                searchSpeed = searchSpeed / 2
            changed = True
            threshold += searchSpeed
        else:
            changed = False
            threshold -= searchSpeed
        # peaks=cleanDoubleStrokes(madmom.features.onsets.peak_picking(superflux_3,threshold),resolution)
        peaks = cleanDoubleStrokes(pick_onsets(H0, delta=threshold), resolution)
    definingBins = findDefBins(peaks, filt_spec)
    return peaks, definingBins, threshold


def liveTake():
    global _ImRunning
    _ImRunning = True
    stompResolution = 1
    buffer = np.zeros(shape=(2277920))  # max take length
    j = 0
    time.sleep(0.1)
    strm = madmom.audio.signal.Stream(sample_rate=SAMPLE_RATE, num_channels=1, frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    for i in strm:
        buffer[j:j + HOP_SIZE] = i[:HOP_SIZE]
        j += HOP_SIZE
        if j >= 2217920 or (not _ImRunning):
            buffer[j:j + 6000] = np.zeros(6000)
            strm.close()
            return buffer[:j + 6000]


def getTempomap(H):
    if H.ndim < 2:
        onsEnv = H
    else:
        onsEnv = np.sum(H, axis=0)
    bdTempo = (librosa.beat.tempo(onset_envelope=onsEnv, sr=SAMPLE_RATE, hop_length=HOP_SIZE,
                                  ac_size=2))#aggregate=None))
    #bdAvg = movingAverage(bdTempo, window=30000)
    #avgTempo = np.mean(bdAvg)
    #return tempoMask(bdAvg), avgTempo
    return bdTempo


def processLiveAudio(liveBuffer=None, peakList=None, Wpre=None, quant_factor=0.0):
    filt_spec = get_preprocessed_spectrogram(liveBuffer)
    iters = 1.

    for i in range(int(iters)):
        # Xpost, W, H = semi_adaptive_NMFB(filt_spec.T,basis,n_iterations=2000,
        #                             alternate=False, adaptive=False, leeseung='bat', div=0, sp=0)
        H = NMFD(filt_spec.T, iters=500, Wpre=Wpre)
        if i == 0:
            WTot, HTot = Wpre, H
        else:

            WTot += Wpre
            HTot += H
    Wpre = (WTot) / iters
    H = (HTot) / iters


    onsets=None
    for i in range(len(peakList)):
        for k in range(K):
            index = i * K + k
            if k == 0:
                H0 = superflux(A=Wpre.T[:, index, :].sum(), B=H[index])
            else:
                H0 = H0 + superflux(A=Wpre.T[:, index, :].sum(), B=H[index])
        if i==0:

            onsets=H0
        else:
            onsets=np.add(onsets,H0)

        peaks = pick_onsets(H0, delta=DELTA)

        # quant_factor > 0:
        #    TEMPO = DEFAULT_TEMPO
        #   qPeaks = timequantize(peaks, avgTempo, TEMPO)
            # qPeaks = quantize(peaks, tempomask, strength=quant_factor, tempo=TEMPO, conform=False)
            # qPeaks=qPeaks*changeFactor
        #else:
        qPeaks = peaks

        # if k==0:
        peakList[i].set_hits(qPeaks)
        # else:
        #    peakList[i].concat_hits(qPeaks)

    # duplicateResolution = 0.05
    # for i in peakList:
    #    precHits = frame_to_time(i.get_hits())
    #    i.set_hits(time_to_frame(cleanDoubleStrokes(precHits, resolution=duplicateResolution)))
    if quant_factor > 0:
        #avgTempo = getTempomap(onsets)
        avgTempo=extract_tempo(onsets)
        for i in peakList:
            i.set_hits(timequantize(i.get_hits(), avgTempo))
    return peakList


# From https://stackoverflow.com/questions/24354279/python-spectral-centroid-for-a-wav-file
def spectral_centroid(x, samplerate=SAMPLE_RATE):
    magnitudes = np.abs(np.fft.rfft(x))  # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0 / samplerate)[:length // 2 + 1])  # positive frequencies
    return np.log(np.sum(magnitudes * freqs) / np.sum(magnitudes))


def ZCR(signal):
    ZC = 0
    for i in range(1, signal.shape[0]):
        if np.sign(signal[i - 1]) != np.sign(signal[i]):
            ZC += 1
    return ZC


# brickwall limiter to even out high peaks
def limitToPercentile(data, limit=90, lowlimit=10, ratio=1):
    limit = np.percentile(data, limit)
    lowlimit = np.percentile(data, lowlimit)
    highPeaks = abs(data) > limit  # Where values higher than the percentile
    data[highPeaks] = limit  # brickwall the signal to the limit
    lowPeaks = abs(data) < lowlimit  # Where values higher than the percentile
    data[lowPeaks] = np.sign(data[lowPeaks]) * lowlimit  # brickwall the signal to the limit
    return (data)


def cleanDoubleStrokes(hitList, resolution=10):
    retList = []
    lastSeenHit = 0
    for i in range(len(hitList)):
        if hitList[i] >= lastSeenHit + resolution:
            retList.append(hitList[i])
            lastSeenHit = hitList[i]
    return (np.array(retList))


##Should i define gap?
def filter_emphasis(spectro, highEmph):
    # disable
    return spectro
    dummy = np.zeros_like(spectro)

    if (highEmph == -1):
        dummy[:, :5] = spectro[:, :5]
    elif (highEmph == 0):
        dummy[:, 2:7] = spectro[:, 2:7]
    elif (highEmph == 1):
        dummy[:, -5:] = spectro[:, -5:]
    elif (highEmph == 2):
        dummy = spectro

    return dummy


def muLaw(Y, mu=10 ** 8):
    # n=frames, i=sub-bands
    x_mu = np.zeros_like(Y)

    for n in range(Y.shape[0]):
        for i in range(Y.shape[1]):
            # x_i_n=Y[n,i].flatten()@Y[n,i].flatten()
            x_i_n = Y[n, i]
            x_mu[n, i] = np.sign(Y[n, i]) * np.log(1 + mu * x_i_n) / np.log(1 + mu)
    return x_mu


def get_preprocessed_spectrogram(buffer):
    buffer = buffer / max(buffer)
    buffer = madmom.audio.FramedSignal(buffer, sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    spec = madmom.audio.spectrogram.FilteredSpectrogram(buffer, filterbank=proc, sample_rate=SAMPLE_RATE,
                                                        frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    # spec=muLaw(spec,mu=10**8)
    spec = spec / spec.max()
    return spec


# Too hard coded method- refine generality
def generate_features(signal, highEmph):
    features = []
    try:
        # fiba=madmom.audio.spectrogram.FilteredSpectrogram(signal,filterbank=proc,sample_rate=44100, f_min=10)
        fiba = get_preprocessed_spectrogram(signal)
        fiba2 = filter_emphasis(fiba, highEmph)
        mfcc2 = madmom.audio.cepstrogram.MFCC(fiba2, num_bands=32)
        mfcc_delta = librosa.feature.delta(mfcc2)
        mfcc_delta2 = librosa.feature.delta(mfcc2, order=2)

        feats = np.append(mfcc2[0], [mfcc2[1]
            , mfcc2[2]
            , mfcc2[3]
            , mfcc_delta[0]
            , mfcc_delta[1]])
        features = (np.append(feats, [np.append([ZCR(signal)]
                                                , [np.append([scipy.stats.kurtosis(signal)]
                                                             , [np.append([scipy.stats.skew(signal)]
                                                                          , [spectral_centroid(signal)])])])]))

    except Exception as e:
        print('feature error:', e)
        """muista panna paddia alkuun ja loppuun"""

    return features


def make_sample(signal, time, n_frames):
    sample = madmom.audio.signal.signal_frame(signal, time, frame_size=n_frames * HOP_SIZE, hop_size=HOP_SIZE, origin=0)
    sample = madmom.audio.signal.normalize(sample)
    return sample


def add_to_samples_and_dictionary(drum, signal, times):
    for i in times:
        sample = make_sample(signal, i, n_frames=4)
        drum.get_samples().append(sample)
        drum.get_templates().append(generate_features(sample, drum.get_highEmph()))


def playSample(data):
    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()
    # open stream (2)
    stream = p.open(format=pyaudio.paFloat32,
                    frames_per_buffer=HOP_SIZE,
                    channels=1,
                    rate=SAMPLE_RATE,
                    output=True)
    # play stream (3)
    f = 0
    # print(len(data))
    while data != '':
        stream.write(data[f])
        f += 1
        if f >= len(data):
            break

    # stop stream (4)
    stream.stop_stream()
    stream.close()

    # close PyAudio (5)
    p.terminate()


# From https://stackoverflow.com/questions/1566936/
class prettyfloat(float):
    def __repr__(self):
        return "%0.3f" % self


# superlux from madmom (Boeck et al)
def superflux(spec_x=[], A=None, B=None):
    """
    Calculate the superflux envelope according to Boeck et al.
    :param spec_x: optional, A Spectrogram the superflux envelope is calculated from, X
    :param A: optional, frequency response of the decomposed spectrogram, W
    :param B: optional, activations of a decomposed spectrogram, H
    :return: Superflux envelope of a spectrogram
    :notes: Must check inputs so that A and B have to be input together
    """
    # if A and B are input the spec_x is recalculated
    if A != None:
        kernel = np.hamming(8)

        B = np.convolve(B, kernel, 'same')
        B = B / max(B)
        spec_x = np.outer(A, B)

    diff = np.zeros_like(spec_x.T)
    size = (2, 1)
    max_spec = maximum_filter(spec_x.T, size=size)
    diff[1:] = (spec_x.T[1:] - max_spec[: -1])
    pos_diff = np.maximum(0, diff)
    sf = np.sum(pos_diff, axis=1)
    sf = sf / max(sf)
    return sf


def frame_to_time(frames, sr=SAMPLE_RATE, hop_length=HOP_SIZE, hops_per_frame=1):
    """
    Transforms frame numbers to time values
    :param frames: list of integers to transform
    :param sr: int, Sample rate of the FFT
    :param hop_length: int, Hop length of FFT
    :param hops_per_frame: ??
    :return: Numpy array of time values
    """
    samples = (np.asanyarray(frames) * (hop_length / hops_per_frame)).astype(int)
    return np.asanyarray(samples) / float(sr)


def time_to_frame(times, sr=SAMPLE_RATE, hop_length=HOP_SIZE, hops_per_frame=1):
    """
    Transforms time values to frame numbers
    :param times: list of timevalues to transform
    :param sr: int, Sample rate of the FFT
    :param hop_length: int, Hop length of FFT
    :param hops_per_frame: ??
    :return: Numpy array of frame numbers
    """
    samples = (np.asanyarray(times) * float(sr))
    return (np.asanyarray(samples) / (hop_length / hops_per_frame)).astype(int)


def f_score(hits, hitNMiss, actual):
    """
    Function to calculate precisionm, recall and f-score
    :param hits: array of true positive hits
    :param hitNMiss: array of all detected hits
    :param actual: array of pre annotated hits
    :return: list of floats (precision, recall, fscore)
    :exception: e if division by zero occurs when no hits are detected, or there are no true hits returns zero values
    """
    try:
        precision = (float(hits) / hitNMiss)
        recall = (float(hits) / actual)
        fscore = (2 * ((precision * recall) / (precision + recall)))
        return (precision, recall, fscore)
    except Exception as e:
        print(e)
        return (0.0, 0.0, 0.0)


def k_in_n(k, n, window=1):
    """
    Helper function to calculate true positive hits for precision, recall and f-score calculation

    :param k: numpy array, list of automatic annotation hits
    :param n: numpy array, list of pre annotated hits
    :param window: float, the windoe in which the hit is regarded as true positive
    :return: float, true positive hits
    """
    hits = 0
    for i in n:
        for j in k:
            if (j - window <= i <= j + window):
                hits += 1
                break
            if (j + window > i):
                break
    return float(hits)

def extract_tempo(onsets=None, window_size_in_s=8):
    # #return
    # bpm_range=range(30,250, 1)
    # #novelty curve
    # #showEnvelope(onsets)
    # #temps=librosa.feature.tempogram(onset_envelope = onsets, sr=SAMPLE_RATE,
    # #hop_length = HOP_SIZE)
    # #tempo=librosa.beat.tempo(onset_envelope=onsets, sr=SAMPLE_RATE,
    # #hop_length = HOP_SIZE)
    # #showEnvelope(onsets[3000:4000])
    # def running_mean(x, N):
    #     cumsum = np.cumsum(np.insert(x, 0, 0))
    #     return (cumsum[N:] - cumsum[:-N]) / float(N)
    #
    # #onsets=onsets[:-99]-running_mean(onsets,100)
    # #onsets = np.array([0 if i < 0 else i for i in onsets])
    # #onsets=onsets/onsets.max()
    # #from Peter Grosche and Meinard Muller
    # def F(Delta_hat):
    #     N = 345
    #     W = scipy.hanning(2 * N + 1)
    #
    #     ftt=[]
    #     for i in range(30, 120):
    #         summa=[]
    #         for t in range(Delta_hat.size):
    #
    #                 summa.append((Delta_hat[n]*W[t%N]*np.exp(-1j * 2 *np.pi*(i/60)*n)).sum())
    #         ftt.append(summa)
    #         print(len(summa))
    #     return np.array(ftt)
    #
    # def tau(Taut):
    #     return np.argmax(Taut)
    #
    # def phit(Taut):
    #     return((1/(2*np.pi))*(Taut.real/np.abs(Taut)))
    #
    # def kernelt_n(W, Taut, t, n):
    #     return (W(n-t)*np.cos(2*np.pi*(taut(Taut)/60*n-phit(Taut))))
    # # def cn(n):
    # #     c = y * np.exp(-1j * 2 * n * np.pi * time / period)
    # #     return c.sum() / c.size
    # #
    # # def f(x, Nh):
    # #     f = np.array([2 * cn(i) * np.exp(1j * 2 * i * np.pi * x / period) for i in range(1, Nh + 1)])
    # #     return f.sum()
    # #print('gfo')
    # #sff=F(onsets)
    # #print(sff.shape)
    # #showFFT(np.abs(sff))
    #
    #
    # #print(sff)
    # win_length = np.asscalar(time_to_frame(4
    #                                        , sr=SAMPLE_RATE,hop_length=HOP_SIZE))
    # stft_bins=int(win_length*2)
    # print(win_length)
    # hop=64
    # #onsets=np.append(onsets,np.zeros(win_length))
    # #onsets = np.append(np.zeros(win_length),onsets)
    # sff = np.abs(librosa.core.stft(onsets, hop_length=hop, win_length=win_length, n_fft=stft_bins,center=True))**2
    # #sff = np.abs(scipy.signal.stft(onsets, fs=1.0, window='hann', nperseg=win_length, noverlap=8, nfft=win_length,
    # #                        detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)[2])
    # print(sff.shape)
    #
    # #showFFT(sff[0:240])
    # #sff2= librosa.feature.tempogram(onset_envelope=onsets, win_length=win_length, hop_length=512)
    # #print(sff2.shape)
    # #showFFT(sff2)
    # #sff3=sff*sff2
    # #showFFT(sff3)
    # #tg=np.mean(sff2,axis=1, keepdims=True)
    # #showEnvelope(tg)
    # #print(tg.shape)
    #
    # tg2 = np.mean(sff,axis=1, keepdims=True)
    # tg2 = tg2.flatten()
    # #showEnvelope(tg2)
    # bin_frequencies = np.zeros(sff.shape[0], dtype=np.float)
    # bin_frequencies[:] = win_length*25.53938/ (np.arange(sff.shape[0]))
    # #prior = np.exp(-0.5 * ((np.log2(bin_frequencies) - np.log2(120)) / 1.) ** 2)
    #
    # #best_period = np.argmax(tg2[10:] * prior[:, np.newaxis], axis=0)+10
    #
    # print(np.argmax(tg2[10:], axis=0)+10)
    # tempi = bin_frequencies[np.argmax(tg2[10:], axis=0)+10]
    # # Wherever the best tempo is index 0, return start_bpm
    # #tempi[best_period == 0] = 120
    # print( tempi)
    # #showEnvelope(tg)
    #
    # #[50]
    # #[103.359375]   #[40]
    #                 #[129.19921875]
    #
    #
    # #tempogram
    # #PLP
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    #onsets=onsets[:-99]-running_mean(onsets,100)
    #onsets = np.array([0 if i < 0 else i for i in onsets])
    #onsets=onsets/onsets.max()
    def F(novelty_curve, win, noverlap, omega, sr):
        """

        :param novelty_curve: onsets
        :param win: window function
        :param noverlap: hops
        :param omega: range of tempos/60
        :param sr: samplerate
        :return: partial stft
        """

        win_len = len(win)
        hopsize = win_len - noverlap
        T = (np.arange(0,(win_len),1)/ sr).T

        win_num = int((novelty_curve.size - noverlap) / (win_len - noverlap))
        #print(novelty_curve.shape,T.shape,win_num, len(omega))
        x = np.zeros((win_num, len(omega)))
        t = np.arange(int(win_len / 2), int(novelty_curve.size - win_len / 2),hopsize)/sr

        tpiT= 2 * np.pi * T

        #for each frequency given in f
        for i in range(omega.size):
            tpift = omega[i]*tpiT
            cosine = np.cos(tpift)
            sine = np.sin(tpift)

            for w in range(win_num):
                start = (w) * hopsize
                stop = start + win_len
                sig = novelty_curve[start:stop]*win
                co = sum(sig* cosine)
                si = sum(sig* sine)
                x[w, i] = (co + 1j * si)

        return t, x.T
    win_len_s=window_size_in_s
    N=int(SAMPLE_RATE/HOP_SIZE*win_len_s)
    W = scipy.hanning(N)
    tempo_div = int(DEFAULT_TEMPO / 2)
    min_bpm=int(tempo_div/2)
    max_bpm=int(tempo_div*3)
    tic=time.clock()
    def squeeze(sff, tempo_target):
        sff2=np.zeros((sff.shape))
        for i in range(sff.shape[1]):
            j=max_bpm-min_bpm-1
            while j>tempo_target*2:
                sff2[int(j/2),i]+=sff[j,i]
                j-=1
            j = 0
            while j < tempo_target/2:
                sff2[int(j *2), i] += sff[j, i]
                j += 1
        return sff2
    #Pad onsets and perform librosa Autocorrelation tempogram
    onsets = np.pad(onsets, int(N//2)+1,
                            mode='mean')
    fonsets=librosa.util.frame(onsets,
                           frame_length=N,
                           hop_length=int(1))[min_bpm:max_bpm,:]
    powspec = np.abs(fft.fft(fonsets, n=2 * fonsets.shape[0] + 1, axis=0)) ** 2
    autocorr = fft.ifft(powspec, axis=0, overwrite_x=True)[:int(powspec.shape[0]/2)]
    sff=autocorr.real
    sff_mean=np.mean(sff, axis=1, keepdims=True)
    #perform librosa tempo extraction with two iterations
    bpms = librosa.core.tempo_frequencies(sff.shape[0], hop_length=HOP_SIZE, sr=SAMPLE_RATE)
    prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(DEFAULT_TEMPO))/1.) ** 2)
    best_period = np.argmax(sff_mean * prior[:, np.newaxis], axis=0)
    #first find the most common tempo of the take
    tempi = bpms[best_period]
    print(tempi)
    #then force the tempi to that area
    prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(tempi)) / 0.5) ** 2)
    best_period = np.argmax(sff * prior[:, np.newaxis], axis=0)
    tempi = bpms[best_period]

    # Wherever the best tempo is index 0, return start_bpm
    tempi[best_period == 0] = min_bpm

    #Smooth the results so we dont get erratic tempoi changes in fills etc.
    tempi=movingAverage(tempi,N)
    #comment out for Without smoothing
    showEnvelope(tempi)


    #Tempo and it's derivatives
    tempos = np.arange(tempo_div, int(tempo_div*3), tempo_div)

    #Change tempi to tempo quantization multipliers
    for i in range(tempi.size):
        tempo = min(tempos, key=lambda x: abs(x - tempi[i]))
        tempi[i] = tempi[i] / tempo

    #print('\ntempomap time:{}'.format(time.clock()-tic))
    return (tempi)

def tempoMask(tempos):
    """
    Create a mask of 16th notes for the duration of the drumtake based on the tempomap
    :param tempos: numpy array of tempos
    :return: list of 16th note indices
    """
    # Move all tempos to half-notes to counter erratic behaviour when tempo extraction doubles tempo value.
    # for i in range(tempos.size):
    #    if tempos[i] > 100:
    #        while tempos[i] > 100:
    #            tempos[i] = tempos[i] / 2
    # define the length of a sixteenthnote in ms in relation to tempo at time t
    sixtLength = MS_IN_MIN / tempos / SXTH_DIV
    # define the length of a frame in ms
    frameLength = 1000 * HOP_SIZE / SAMPLE_RATE
    # extract indices in X that correspond to a 16th note in tempo at time t
    # by calculating modulo and taking the inverse boolean array mod(a,b)==0 ==True
    indices = np.array([int((s % (sixtLength[s] / frameLength))) for s in range(0, tempos.shape[0])])
    invertedIndices = np.invert(indices.astype('bool'))
    return invertedIndices


def timequantize(X, tempomap):
    """
    Quantizes the hits X according to a tempomap
    :param X: numpy array, The onsets to quantize
    :param tempomap: numpy array, the tempo modifiers for each frame
    :return: numpy array, quantized onsets

    Notes:
    """
    retX=np.zeros((X.size))
    newgap = sum(tempomap[:X[0]])
    retX[0]=newgap
    for i in range(1,X.size):
            #Calculate the gap between two consecutive hits
            newgap=sum(tempomap[X[i-1]:X[i]])
            #move the hit to last hit+newgap
            retX[i] =retX[i-1]+newgap
    return retX


def quantize(X, mask, strength=1, tempo=DEFAULT_TEMPO, conform=False):
    """
    Quantize hits accordng to 16th Note mask
    :param X: np.array List of Drumhits
    :param mask: np.array Precalculated tempo mask of performance
    :param strength:, float [0,1] Strength of quantization
    :param tempo: Int, Tempo to quantize to
    :param conform: Boolean, Weather to quantize to tempo or just to mask
    :return: numpy array of quantized drumhits
    """
    # Create a mask of constant tempo
    if conform:
        ####
        # drop less than 100 to half
        # if tempo < 100:
        #    tempo = tempo / 2
        # A mask with True at 16th notes
        conformMask = tempoMask(np.full(mask.size * 2, tempo))
        # The tempomask of drumtake
        trueInd = np.where(mask == True)[0]
        # Shorter of the masks
        dim = min(np.where(conformMask == True)[0].size, trueInd.size)
        # Shift length to nearest true conformMask index(backwards)
        cMask = np.where(conformMask == True)[0][:dim] - trueInd[:dim]
        # Mask to store shift values
        shiftMask = mask.astype(int)
        # Store shift lengths to shiftMask for lookup when conforming to tempo
        n = 0
        for i in trueInd:
            shiftMask[i] = cMask[n]
            n += 1
    # iterate found onsets
    retX = np.zeros_like(X)
    n = 0
    for i in X:
        k = 0
        notfound = True

        # shift
        j = 0

        # set limit to shift
        jLim = min(i, mask.size - i)
        while notfound:

            # If the shift is at either end of the mask remove onset
            if j == jLim:
                notfound = False
                k = None

            # If the onset is in 16th note mask
            if mask[i] == True:
                notfound = False
                k = i
                if conform:
                    k += shiftMask[i]

            # Move the onset forvard by j frames and compare,
            # if 16th note found in the mask move the onset there
            elif mask[i + j] == True:
                notfound = False
                k = i + j
                if conform:
                    k += shiftMask[i + j]

            # backward move of the onset
            elif mask[i - j] == True:
                notfound = False
                k = i - j
                if conform:
                    k += shiftMask[i - j]

            # increase shift
            j += 1
        # Store the quantized value to return list
        retX[n] = int((k * strength + i * (1 - strength)))
        # increase return list index
        n += 1
        # retX=retX[retX != np.array(None)]
    return retX[retX != 0]


def get_cmask(tempo, tempomask):
    pass


# aloita ikkuna x[:-window]
def movingAverage(x, window=500):
    return median_filter(x, size=(window))


def pick_onsets(F, delta=0.):
    """
    Simple onset peak picking algorithm, picks local maxima that are
    greater than median of local maxima + correction factor.

    :param F: numpy array, Detection Function
    :param delta: float, threshold correction factor
    :return: numpy array, peak indices
    """
    # Indices of local maxima in F
    localMaximaInd = argrelmax(F, order=1)

    # Values of local maxima in F
    localMaxima = F[localMaximaInd[0]]

    # Pick local maxima greater than threshold
    threshold = delta + np.median(localMaxima)
    onsets = np.array(localMaxima >= threshold)
    rets = localMaximaInd[0][onsets]
    # remove peak if detFunc has not been under threshold.
    i = 0
    while i in range(len(rets) - 1):
        if F[rets[i]:rets[i + 1]].min() >= threshold:
            rets = np.delete(rets, i + 1)
        else:
            i += 1
    # Return onset indices
    return rets

    ####Dynamic threshold testing.
    # Not worth the processing time when using superflux envelope for separated audio
    # #Size of local maxima array
    # arSize=F.shape[0]
    # #dynamic threshold arrays
    # T1, T2, Tdyn=np.zeros(arSize),np.zeros(arSize),np.zeros(arSize)
    # #calculate every threshold
    # #n = movingAverage(F, window=23)
    # for i in range(len(F)):
    #     #n[i]=np.median(F[i:i+10])
    #     #n[i] = np.percentile(F[i:i+10],50)
    #     T1[i] = delta + lamb * (np.percentile(F[i:i+10],75) - np.percentile(F[i:i+10],25)) + np.percentile(F[i:i+10],50)
    #     T2[i] = delta * np.percentile(F[i:i+10],100)
    #     p = 2
    #     # final Dyn threshold from battenberg
    #     Tdyn[i]=((T1[i]**p+T2[i]**p)/2.)**(1./p)
    # #nPerc = np.percentile(localMaxima, [25, 50, 75,100])
    # if pics:
    #     showEnvelope(F)
    # if pics:
    #     showEnvelope(T1)
    # #first dyn threshold
    # #T1 = delta+lamb*(nPerc[2]-nPerc[0]) + nPerc[1]
    # #second dyn threshold
    # #T2 =delta*nPerc[3]
    # #Soft maximum variable
    # #p=4
    # #final Dyn threshold from battenberg
    # #Tdyn=((T1**p+T2**p)/2.)**(1./p)
    # #Create a boolean array of indices where local maxima is above Tdyn
    # onsets=np.array(localMaxima>=T1[localMaximaInd[0]])


# Turha
def HFC_filter(X=[], A=None, B=None):
    if A != None:
        kernel = np.hamming(8)

        B = convolve(B, kernel, 'same')
        B = B / max(B)
        X = np.outer(A, B)

    # b, a = signal.butter(4,0.5)
    Xi = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        Xi += i * 100 * X[i]
    plt.figure()
    plt.plot(Xi)
    return (Xi / Xi.max())


import sys


def NMFD(X, iters=50, Wpre=[]):
    """
    :param :
       - X : The spectrogram
       - iters : # of iterations
       - Wpre : Prior vectors of W

    :return :
       - W : unchanged prior vectors of W
       - H : Activations of different drums in X

    """
    # epsilon, added to matrix elements to ensure non-negativity in matrices
    eps = 10 ** -18

    # data size
    M, N = X.shape
    W = Wpre
    M, R, T = W.shape

    # Initial H, non negative, non zero. any non zero value works
    H = np.full((R, N),.5)


    #Make sure the input is normalized
    W = W / W.max()
    X = X / X.max()
    repMat = np.ones((M, N))
    Lambda = np.zeros((M, N))

    err = np.zeros(iters)

    # KLDivergence for cost calculation
    def KLDiv(x, y, LambHat):
        return (x * np.log(LambHat) - x + y).sum()

    def ISDiv(x):
        return (x-np.log(x)-1).sum()

    def computeConv(A, B):
        ABC = np.zeros((B.shape[0], A.shape[1]))
        for m in range(M):
            for r in range(R):
                ABC[m, :] += np.convolve(B[m, r, :], A[r, :])[0:ABC.shape[1]]
        return ABC

    Runner = 1
    mu = 0.275 #sparsity
    beta=1
    for i in range(iters):
        if REQUEST_RESULT:
            return H
        print('\rNMFD iter: %d' % (i + 1), end="", flush=True)

        if i == 0:
            Lambda = eps+computeConv(H, W)
            LambHat = X / (Lambda+eps)

        if Runner == 0:
            shifter = np.zeros((M, N + T))
            #shifter[:, :-T] = LambHat
            shifter[:, :-T] = X*Lambda**(beta-2)
            Hhat = np.zeros((R, N))
            for t in range(T):
                Wt = W[:, :, t].T
                Hhat += Wt @ ((shifter[:, t: t + N])) / (Wt @ (Lambda**(beta-1))+eps)
            H = H * Hhat

        if Runner == 1:
            # From https://github.com/romi1502/NMF-matlab
            # Copyright (C) 2015 Romain Hennequin
            # Seems to improve results a little! But is a little slower
            # This seems to be due to the fact that each frequency sub band is convoluted separately,
            # not matrix multiplied as a whole 1d vs 2d convolution.
            Hu = np.zeros((R, N))
            Hd = np.zeros((R, N))
            for r in range(R):
                for m in range(M):
                    Hu[r, :] += np.convolve(LambHat[m, :], np.flipud(W[m, r, :]))[T - 1:]
                    Hd[r, :] += np.convolve(repMat[m, :], np.flipud(W[m, r, :]))[T - 1:]
            H = H * Hu / (Hd+mu)

        # precompute for error and next round
        Lambda = computeConv(H, W)
        LambHat = X /  (Lambda+eps)

        # 2*eps to ensure the eps don't cancel out each other in the division.
        err[i] = KLDiv(X + eps, Lambda + 2 * (eps), LambHat+eps)
        #err[i] = ISDiv(LambHat)

        if (i >= 1):
            errDiff = (abs(err[i] - err[i - 1]) / (err[1] - err[i]+eps))
            #print(errDiff)
            if errDiff < 0.0005 or err[i] > err[i - 1] :
                break
    return H
import matplotlib.pyplot as plt

def showEnvelope(env):
    """
    Plots an envelope
    i.e. an onset envelope or some other 2d data
    :param env: the data to plot
    :return: None
    """
    plt.figure()
    plt.plot(env)


def showFFT(env):

    """
    Plots a spectrogram

    :param env: the data to plot
    :return: None
    """
    plt.figure()
    plt.imshow(env, aspect='auto', origin='lower')


def acceptHit(value, hits):
    """
    Helper method to clear mistakes of the annotation such as ride crach and an open hi hat.
    :param value: int the hit were encoding
    :param hits: binary string, The hits already found in that location
    :return: boolean, if it is ok to annotate this drum here.
    """
    # Tähän pitää tehä se et vaan vaikka 3 iskua yhteen frameen!
    # Pitää vielä testata

    sum = 0
    for i in range(nrOfDrums):
        if np.bitwise_and(i, hits):
            sum += 1
        if sum > 3:
            return False
    ##Voiko tän jättää pois??
    offendingHits = np.array([
        value == 2 and np.bitwise_and(3, hits),
        value == 2 and np.bitwise_and(8, hits),
        value == 3 and np.bitwise_and(2, hits),
        value == 3 and np.bitwise_and(7, hits),
        value == 6 and np.bitwise_and(7, hits),
        value == 6 and np.bitwise_and(3, hits),
        value == 6 and np.bitwise_and(2, hits),
        value == 7 and np.bitwise_and(6, hits),
        value == 8 and np.bitwise_and(2, hits),
        value == 8 and np.bitwise_and(3, hits)
    ])
    if offendingHits.any():
        return False
    return True


def truncZeros(frames):
    zeros = 0
    for i in range((frames.size)):
        if frames[i] == 0:
            zeros += 1
            # longest pause
            # if zeros == 4:
            #    frames[i - zeros + 1] = -zeros
            #    zeros = 0
            #    continue
        elif zeros != 0 and frames[i] != 0:
            # Encode pause to a negative integer
            frames[i - zeros] = -zeros
            zeros = 0
    # 0:1, 1:2, 2:3, 3:4 ->
    frames = frames[frames != 0]
    return frames


def mergerowsandencode(a):
    """
        Merges hits occuring at the same frame.
        i.e. A kick drum hit  and a closed hihat hit at
        frame 100 are encoded from
            100 0
            100 2
        to
            100 5
        where 5 is decoded into char array 000000101 in the GRU-NN.

        Also the hits are assumed to be in tempo 120bpm and all the
        frames not in 120bpm 16th notes are quantized.

        :param a: numpy array of hits

        :return: numpy array of merged hits

        Notes
        -----
        The tempo is assumed to be the global value

        """
    # Sixteenth notes length in this sample rate and frame size
    sixtDivider = SAMPLE_RATE / FRAME_SIZE / SXTH_DIV
    if QUANTIZE:
        for i in range(len(a)):
            # Move frames to nearest sixteenth note
            a[i] = (np.rint(a[i][0] / sixtDivider),) + a[i][1:]
    # Define max frames from the first detected hit to end
    maxFrame = int(a[-1][0] - a[0][0] + 1)
    #print(maxFrame)
    # spaceholder for return array
    frames = np.zeros(maxFrame, dtype=int)
    for i in range(len(a)):
        # define true index by substracting the leading empty frames
        index = int(a[i][0] - a[0][0])
        # The actual hit information
        value = a[i][1]
        # Encode the hit into a charachter array, place 1 on the index of the drum #
        if acceptHit(value, frames[index]):
            frames[index] = np.bitwise_or(frames[index], 2 ** value)

    # return array of merged hits starting from the first occurring hit event
    if ENCODE_PAUSE:
        frames = truncZeros(frames)
    return frames


def splitrowsanddecode(a):
    """
        Split hits occuring at the same frame to separate lines containing one row per hit.
        i.e. A kick drum hit  and a closed hihat hit at
        frame 100 are decoded from
            100 5
        to
            100 0
            100 2

        :param a: numpy array of hits

        :return: numpy array of separated hits

        """
    decodedFrames = []
    # multiplier to make tempo the global tempo after generation.

    frameMul = 1
    if QUANTIZE:
        frameMul = SAMPLE_RATE / FRAME_SIZE / SXTH_DIV
    i = 0
    pause = 0
    while i in range(len(a)):
        # if we find a pause we add that to the pause offset
        if ENCODE_PAUSE:
            if a[i] < 0:
                pause += (-1 * a[i]) - 1
                i += 1
                continue
        # split integer values to a binary array
        for j, k in enumerate(dec_to_binary(a[i])):
            if int(k) == 1:
                # store framenumber(index) and drum name to list, (nrOfDrums-1) to flip drums to right places,
                decodedFrames.append([(i + pause) * frameMul, abs(j - (nrOfDrums - 1))])
        i += 1
    # return the split hits
    return decodedFrames


def dec_to_binary(f):
    """
        Returns a binary representation on a given integer

        :param  f: an integer
        :return: A binary array representation of f


            """
    return format(f, "0{}b".format(nrOfDrums))
