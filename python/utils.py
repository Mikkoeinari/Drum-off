import time
#import sys
import librosa
import madmom
import numpy as np
import pandas as pd
import scipy
from scipy import fftpack as fft
from scipy.ndimage.filters import median_filter, maximum_filter
from scipy.signal import argrelmax, stft
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.utils import resample

FRAME_SIZE = 2 ** 11
HOP_SIZE = 2 ** 9
# After quantization to 120bpm we use hop size 353 ~ the resolution of 32th note at 120bpm
Q_HOP =  353
SAMPLE_RATE = 44100
FREQUENCY_PRE = np.ones((24))  # [0,16384]#0-2^14
MIDINOTE = 36  # kickdrum in most systems
THRESHOLD = 0.0
PROBABILITY_THRESHOLD = 0.0
DEFAULT_TEMPO = 120  # 117.1875
DRUMKIT_PATH = '../trainSamplet/'
REQUEST_RESULT = False
DELTA = 0.15
midinotes = [36, 38, 42, 46, 50, 43, 51, 49, 44]  # BD, SN, CHH, OHH, TT, FT, RD, CR, SHH, Here we need generality
nrOfDrums = 24  # Maximum kit size
nrOfPeaks = 16  #IF CHANGED ALL PREVIOUS SOUNDCHECKS INVALIDATE!!!
max_n_frames = 10
total_priors = 0
MS_IN_MIN = 60000
SXTH_DIV = 16
QUANTIZE = False
ENCODE_PAUSE = True
_ImRunning = False
# Shortest processing time good results
proc = madmom.audio.filters.BarkFilterbank(
    madmom.audio.stft.fft_frequencies(num_fft_bins=int(FRAME_SIZE / 2), sample_rate=SAMPLE_RATE),
    num_bands='double', fmin=20.0, fmax=15500.0, norm_filters=True, unique_filters=True)
# Best result-longest processing time
# proc =madmom.audio.filters.MelFilterbank(
#    madmom.audio.stft.fft_frequencies(num_fft_bins=int(FRAME_SIZE / 2), sample_rate=SAMPLE_RATE),
# num_bands=128, fmin=20.0, fmax=17000.0, norm_filters=True, unique_filters=True)
# second best in everything
# proc =madmom.audio.filters.LogarithmicFilterbank(
#    madmom.audio.stft.fft_frequencies(num_fft_bins=int(FRAME_SIZE / 2), sample_rate=SAMPLE_RATE),
#    num_bands=18, fmin=20.0, fmax=17000.0, fref=110.0, norm_filters=True, unique_filters=True, bands_per_octave=True)

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

    def __init__(self, name, highEmph, peaks, tails=None, heads=None, frequency_pre=FREQUENCY_PRE,
                 midinote=MIDINOTE, threshold=THRESHOLD, probability_threshold=PROBABILITY_THRESHOLD
                 , hitlist=None, **kwargs):

        # set attributes
        self.name = name
        self.highEmph = highEmph
        self.peaks = peaks
        if len(frequency_pre):
            self.frequency_pre = frequency_pre
        if len(tails):
            self.tails = tails
        if len(heads):
            self.heads = heads
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

    def set_heads(self, heads):
        self.heads = heads

    def get_heads(self):
        return self.heads

    def set_tails(self, tails):
        self.tails = tails

    def get_tails(self):
        return self.tails

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


def findDefBinsBG(frames, filteredSpec, ConvFrames, K, bs_len=32):
    """
    Calculate the prior vectors for W to use in NMF
    :param frames: Numpy array of hit locations (frame numbers)
    :param filteredSpec: Spectrogram, the spectrogram where the vectors are extracted from
    :return: tuple of Numpy arrays, prior vectors Wpre,heads for actual hits and tails for decay part of the sound
    """
    global total_priors
    gaps = np.zeros((frames.shape[0], max_n_frames))
    # gaps = np.zeros((frames.shape[0], ConvFrames))
    for i in range(frames.shape[0]):
        for j in range(ConvFrames):
            gaps[i, j] = frames[i] + j

    a = np.reshape(filteredSpec[gaps.astype(int)], (nrOfPeaks, -1))
    a2=a
    #a2=resample(a, n_samples=bs_len, replace=True, random_state=2)
    bgMeans = BayesianGaussianMixture(
        n_components=K, covariance_type='spherical',  # weight_concentration_prior=1,
        weight_concentration_prior_type='dirichlet_process',
        mean_precision_prior=.8,  # covariance_prior=np.eye(proc.shape[1]*ConvFrames),
        init_params="random", max_iter=100, random_state=2).fit(a2)
    K1 = np.unique(bgMeans.predict(a2))
    #print(K1)
    heads = np.zeros((proc.shape[1], max_n_frames, K1.shape[0]))
    # heads = np.zeros((proc.shape[1], ConvFrames, K1.shape[0]))
    for i in range(K1.shape[0]):
        heads[:, :, i] = np.reshape(bgMeans.means_[K1[i], :], (proc.shape[1], max_n_frames), order='F')

        # heads[:, :, i] = np.reshape(bgMeans.means_[K1[i], :], (proc.shape[1], ConvFrames), order='F')
    #
    tailgaps = np.zeros((frames.shape[0], max_n_frames))

    # tailgaps = np.zeros((frames.shape[0], ConvFrames))
    for i in range(frames.shape[0]):
        for j in range(tailgaps.shape[1]):
            tailgaps[i, j] = frames[i] + j + ConvFrames
    a = np.reshape(filteredSpec[tailgaps.astype(int)], (nrOfPeaks, -1))
    a2=a
    #a2 = resample(a, n_samples=bs_len, replace=True, random_state=2)
    bgMeans = BayesianGaussianMixture(
        n_components=K, covariance_type='spherical',  # weight_concentration_prior=1,
        weight_concentration_prior_type='dirichlet_distribution',
        mean_precision_prior=0.8,  # covariance_prior=np.eye(proc.shape[1]*ConvFrames),
        init_params="random", max_iter=100, random_state=2).fit(a2)
    K2 = np.unique(bgMeans.predict(a2))
    tails = np.zeros((proc.shape[1], max_n_frames, K2.shape[0]))
    # tails = np.zeros((proc.shape[1], ConvFrames, K2.shape[0]))
    for i in range(K2.shape[0]):
        tails[:, :, i] = np.reshape(bgMeans.means_[K2[i], :], (proc.shape[1], max_n_frames), order='F')

        # tails[:, :, i] = np.reshape(bgMeans.means_[K2[i],:], (proc.shape[1], ConvFrames), order='F')
    total_priors += K1.shape[0] + K2.shape[0]
    return (heads, tails, K1, K2)
# from itertools import combinations
# def em_mdl(templates):
#     score=0
#     temp_temps=[]
#     #iterate
#     def ISDiv(x,y):
#         return (y/x - np.log(y/x) - 1).sum()
#     def join_nearest(clusters):
#         n=clusters.shape[0]
#         indexs=combinations(range(clusters.shape[0]),2)
#         divergences=np.zeros(indexs.shape[0])
#         for i in range(indexs.shape[0]):
#             divergences[i]=ISDiv(clusters[indexs[i][0]], clusters[indexs[i][1]])
#
#     for i in range(templates.shape[0]):
#         #calculate and store mdl for i
#         current_score=mdl(current)
#         if current_score<=score:
#             score=current_score
#             temp_temps=current
#         #join two nearest clusters
#         current=join_nearest(current)
#
#
# def findDefBins(frames, filteredSpec, ConvFrames, K):
#     """
#     Calculate the prior vectors for W to use in NMF
#     :param frames: Numpy array of hit locations (frame numbers)
#     :param filteredSpec: Spectrogram, the spectrogram where the vectors are extracted from
#     :return: tuple of Numpy arrays, prior vectors Wpre,heads for actual hits and tails for decay part of the sound
#      """
#     global total_priors
#     gaps = np.zeros((frames.shape[0], ConvFrames))
#     for i in range(frames.shape[0]):
#         for j in range(gaps.shape[1]):
#             gaps[i, j] = frames[i] + j
#
#     a = np.reshape(filteredSpec[gaps.astype(int)], (nrOfPeaks, -1))
#     kmeans = KMeans(n_clusters=K).fit(a)
#
#     heads = np.zeros((proc.shape[1], ConvFrames, K))
#     for i in range(K):
#         heads[:, :, i] = np.reshape(kmeans.cluster_centers_[i], (proc.shape[1], ConvFrames), order='F')
#     heads = em_mdl(heads)
#     tailgaps = np.zeros((frames.shape[0], ConvFrames))
#     for i in range(frames.shape[0]):
#         for j in range(gaps.shape[1]):
#             tailgaps[i, j] = frames[i] + j + ConvFrames
#
#     a = np.reshape(filteredSpec[tailgaps.astype(int)], (nrOfPeaks, -1))
#     kmeans = KMeans(n_clusters=K).fit(a)
#
#     tails = np.zeros((proc.shape[1], ConvFrames, K))
#     for i in range(K):
#         tails[:, :, i] = np.reshape(kmeans.cluster_centers_[i], (proc.shape[1], ConvFrames), order='F')
#
#     tails=em_mdl(tails)
#     total_priors = heads.shape[0]+tails.shape[0]
#
#     return (heads, tails, 0, 0)


def getStompTemplate(numHits=2, convFrames=10):
    """

    :param numHits:
    :return:
    """
    global _ImRunning

    _ImRunning = True
    stompResolution = 4
    buffer = np.zeros(shape=(2646000))
    j = 0
    time.sleep(0.1)
    strm = madmom.audio.signal.Stream(sample_rate=SAMPLE_RATE, num_channels=1, frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    for i in strm:
        # print(i.shape)
        buffer[j:j + HOP_SIZE] = i[:HOP_SIZE]
        j += HOP_SIZE
        if j >= 2646000 or (not _ImRunning):
            buffer[j:j + 6000] = np.zeros(6000)
            strm.close()
            return buffer[:j + 6000]


def getPeaksFromBuffer(filt_spec, resolution, numHits, convFrames, K):
    threshold = 1
    searchSpeed = .1
    # peaks=cleanDoubleStrokes(madmom.features.onsets.peak_picking(superflux_3,threshold),resolution)
    H0 = superflux(spec_x=filt_spec.T)

    peaks = cleanDoubleStrokes(pick_onsets(H0 / H0.max(), delta=threshold), resolution)
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
    return peaks


def recalculate_thresholds(filt_spec, shifts, drums, drumwise=False, rm_win=3):
    onset_alg = 0
    total_heads = 0
    Wpre = np.zeros((proc.shape[1], total_priors, max_n_frames))
    for i in range(len(drums)):
        heads = drums[i].get_heads()
        K1 = heads.shape[2]
        ind = total_heads
        for j in range(K1):
            Wpre[:, ind + j, :] = heads[:, :, j]
            total_heads += 1
    total_tails = 0
    for i in range(len(drums)):
        tails = drums[i].get_tails()
        K2 = tails.shape[2]
        ind = total_heads + total_tails
        for j in range(K2):
            Wpre[:, ind + j, :] = tails[:, :, j]
            total_tails += 1

    H,Wpre, err1 = NMFD(filt_spec.T, iters=128, Wpre=Wpre, include_priors=False)
    total_heads = 0
    Hs = []
    for i in range(len(drums)):
        heads = drums[i].get_heads()
        K1 = heads.shape[2]
        ind = total_heads
        if onset_alg == 0:
            for k in range(K1):
                index = ind + k
                HN = superflux(A=Wpre.T[:, index, :].sum(), B=H[index], win_size=8)
                #HN = HN / HN.max()
                if k == 0:
                    H0 = HN
                else:
                    H0 = np.maximum(H0, HN)
                total_heads += 1
        elif onset_alg==1:
            for k in range(K1):
                index = ind + k
                HN=get_preprocessed_spectrogram(A=Wpre.T[:, index, :].sum(), B=H[index], test=True)[:,0]
                if k == 0:
                    H0 = HN
                else:
                    H0 = np.maximum(H0, HN)
                total_heads += 1
                H0 = H0 / H0.max()
        else:
            kernel = np.hanning(8)
            for k in range(K1):
                index = ind + k
                HN = H[index]
                HN = np.convolve(HN, kernel, 'same')
                HN = HN / HN.max()
                if k == 0:
                    H0 = HN
                else:
                    H0 = np.maximum(H0, HN)
                total_heads += 1
            # H0 = H0 / H0.max()
        #H0 = H0[:-(rm_win - 1)] - running_mean(H0, rm_win)
        #H0 = np.array([0 if i < 0 else i for i in H0])
        H0 = H0 / H0.max()
        Hs.append(H0)

        besthits = []
        if drumwise:
            deltas = np.linspace(0.15, 1, 10)
            f_zero = 0
            threshold = 0
            for d in deltas:
                if i<len(shifts)-1:
                    peaks = pick_onsets(H0[shifts[i]:shifts[i+1]], delta=d)
                    #showEnvelope(H0[shifts[i]:shifts[i+1]])
                else:
                    peaks = pick_onsets(H0[shifts[i]:], delta=d)
                #print(peaks.shape[0])

                drums[i].set_hits(peaks[np.where(peaks < filt_spec.shape[0] - 1)])
                predHits = drums[i].get_hits()
                actHits = drums[i].get_peaks()
                trueHits = k_in_n(actHits, predHits, window=1)
                prec, rec, f_drum = f_score(trueHits, predHits.shape[0], actHits.shape[0])
                #print(d,f_drum)
                if f_drum > f_zero:
                    f_zero = f_drum
                    threshold = d
            print('delta:', threshold, f_zero)
            drums[i].set_threshold(threshold)

    if not drumwise:
        # print(len(Hs))
        deltas = np.linspace(0, 1, 100)
        f_zero = 0
        threshold = 0
        for d in deltas:
            precision, recall, fscore, true_tot = 0, 0, 0, 0
            for i in range(len(drums)):
                peaks = pick_onsets(Hs[i], delta=d)
                drums[i].set_hits(peaks[np.where(peaks < filt_spec.shape[0] - 1)])
                predHits = drums[i].get_hits()
                # print(predHits)
                actHits = drums[i].get_peaks() + shifts[i] - 1
                # print(actHits)
                trueHits = k_in_n(actHits, predHits, window=1)
                prec, rec, f_drum = f_score(trueHits, predHits.shape[0], actHits.shape[0])
                precision += prec * actHits.shape[0]
                recall += rec * actHits.shape[0]
                fscore += (f_drum * actHits.shape[0])
                true_tot += actHits.shape[0]
            if (fscore / true_tot) > f_zero:
                f_zero = (fscore / true_tot)
                threshold = d
        print('delta:', threshold, f_zero)
        for i in range(len(drums)):
            drums[i].set_threshold(threshold)
            #fix delta
            #drums[i].set_threshold(0.16)
            # Try loosening plate drum thresholds: NOT FINAL!!!
            if i in [2, 3, 6,7,8]:
                pass
                #drums[i].set_threshold(threshold)

    # mean = sum(drums[i].get_threshold() for i in range(len(drums))) / len(drums)
    # for i in range(len(drums)):
    #    drums[i].set_threshold(mean)


def liveTake():
    global _ImRunning
    _ImRunning = True
    stompResolution = 1
    buffer = np.zeros(shape=(44100*10+18000))  # max take length (30s.)

    j = 0
    time.sleep(0.1)
    strm = madmom.audio.signal.Stream(sample_rate=SAMPLE_RATE, num_channels=1, frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    for i in strm:
        buffer[j:j + HOP_SIZE] = i[:HOP_SIZE]
        j += HOP_SIZE
        if j >=buffer.shape[0]-18000 or (not _ImRunning):
            buffer[j:j + 6000] = np.zeros(6000)
            strm.close()
            #Should this yield instead of returning? To record as long as the drummer wants...
            return buffer[:j + 6000]


def processLiveAudio(liveBuffer=None, drums=None, quant_factor=0.0, iters=0, method='NMFD',thresholdAdj=0.0):
    onset_alg = 0
    filt_spec = get_preprocessed_spectrogram(liveBuffer, sm_win=4)
    stacks = 1.
    total_priors = 0
    for i in range(len(drums)):
        total_priors += drums[i].get_heads().shape[2]
        total_priors += drums[i].get_tails().shape[2]
    Wpre = np.zeros((proc.shape[1], total_priors, max_n_frames))
    total_heads = 0
    for i in range(len(drums)):
        heads = drums[i].get_heads()
        K1 = heads.shape[2]
        ind = total_heads
        for j in range(K1):
            Wpre[:, ind + j, :] = heads[:, :, j]
            total_heads += 1
    total_tails = 0

    for i in range(len(drums)):
        tails = drums[i].get_tails()
        K2 = tails.shape[2]
        ind = total_heads + total_tails
        for j in range(K2):
            Wpre[:, ind + j, :] = tails[:, :, j]
            total_tails += 1
    # Wpre = Wpre[:, :total_heads + total_tails, :]
    #PSA(filt_spec[:1500], Wpre[:,:9,0])
    for i in range(int(stacks)):
        if method == 'NMFD' or method == 'ALL':
            H,Wpre, err1 = NMFD(filt_spec.T, iters=iters, Wpre=Wpre, include_priors=True)
        if method == 'NMF' or method == 'ALL':
            H, err2 = semi_adaptive_NMFB(filt_spec.T, Wpre=Wpre, iters=iters)
        if method == 'ALL':
            errors = np.zeros((err1.size, 2))
            errors[:, 0] = err1
            errors[:, 1] = err2
            #showEnvelope(errors, ('NMFD Error', 'NMF Error'), ('iterations', 'error'))
        if i == 0:
            WTot, HTot = Wpre, H
        else:

            WTot += Wpre
            HTot += H
    Wpre = (WTot) / stacks
    H = (HTot) / stacks

    onsets = np.zeros(H[0].shape[0])
    total_heads = 0
    picContent=[]
    allPeaks=[]
    for i in range(len(drums)):
        #if i<=9:
        #    showEnvelope(H[i][:1500])
        heads = drums[i].get_heads()
        K1 = heads.shape[2]
        ind = total_heads
        if onset_alg == 0:
            for k in range(K1):
                index = ind + k
                HN = superflux(A=Wpre.T[:, index, :].sum(), B=H[index],win_size=8)
                #HN = HN / HN.max()
                if k == 0:
                    H0 = HN
                else:
                    H0 = np.maximum(H0, HN)
                total_heads += 1
        elif onset_alg==1:
            for k in range(K1):
                index = ind + k
                HN=get_preprocessed_spectrogram(A=Wpre.T[:, index, :].sum(), B=H[index], test=True)[:,0]
                if k == 0:
                    H0 = HN
                else:
                    H0 = np.maximum(H0, HN)
                total_heads += 1
                H0 = H0 / H0.max()
        else:
            kernel = np.hanning(8)
            for k in range(K1):
                index = ind + k
                HN = H[index]
                HN = np.convolve(HN, kernel, 'same')
                HN = HN / HN.max()
                if k == 0:
                    H0 = HN
                else:
                    H0 = np.maximum(H0, HN)
                total_heads += 1
        if i == 0:
            onsets = H0
        else:
            onsets = onsets+H0
        #H0 = H0[:-(rm_win-1)] - running_mean(H0, rm_win)
        #H0 = np.array([0 if i < 0 else i for i in H0])
        #H0=H0/H0.max()

        #showEnvelope(H0)
        peaks = pick_onsets(H0, delta=drums[i].get_threshold()+thresholdAdj)
        if i in [0,1,2]:
            picContent.append([H0[:], pick_onsets(H0[:], delta=drums[i].get_threshold())])
            kernel = np.hanning(8)
        # remove extrahits used to level peak picking algorithm:
        peaks = peaks[np.where(peaks < filt_spec.shape[0] - 1)]
        drums[i].set_hits(peaks)
        # onsets[peaks] = 1
        # quant_factor > 0:
        #    TEMPO = DEFAULT_TEMPO
        #   qPeaks = timequantize(peaks, avgTempo, TEMPO)
        # qPeaks = quantize(peaks, tempomask, strength=quant_factor, tempo=TEMPO, conform=False)
        # qPeaks=qPeaks*changeFactor
        # else:
        allPeaks.extend(peaks)
    #sanity check
    if False:
        # detect peaks in the full spectrogram, compare to detection results and inset peak where none is found
        sanityspec = get_preprocessed_spectrogram(liveBuffer, sm_win=8, test=True)
        # H0=superflux(spec_x=filtspec.T, win_size=8)
        HS = sanityspec[:, 0]
        HS = HS / HS[3:].max()
        sanitypeaks = pick_onsets(HS, delta=0.035)
        for i in sanitypeaks:
            if np.argwhere(allPeaks==i) is None:
                print ('NMFD missed an onset at:', i)

    #showEnvelope(picContent)
    # duplicateResolution = 0.01
    # for i in drums:
    #   precHits = frame_to_time(i.get_hits())
    #   i.set_hits(time_to_frame(cleanDoubleStrokes(precHits, resolution=duplicateResolution)))

    if quant_factor > 0:
        onsets = onsets / onsets.max()
        deltaTempo = extract_tempo(onsets, constant_tempo=True)
        for i in drums:
            hits = i.get_hits()
            if len(hits) is None:
                i.set_hits([])
            else:
                i.set_hits(conform_time(i.get_hits(), deltaTempo, quantize=True))
        print(np.mean(deltaTempo))
        return drums, np.mean(deltaTempo)
    else:
        return drums, 1.0




def get_preprocessed_spectrogram(buffer=None,A=None,B=None, sm_win=4, test=False):
    if buffer is not None:
        buffer = madmom.audio.FramedSignal(buffer, sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
        spec = madmom.audio.spectrogram.FilteredSpectrogram(buffer, filterbank=proc, sample_rate=SAMPLE_RATE,
                                                            frame_size=FRAME_SIZE, hop_size=HOP_SIZE,fmin=20, fmax=17000)

    if A != None:
        spec = np.outer(A, B).T
    #kernel=np.kaiser(6,5)
    if test:
        mu =0.4
        for i in range(spec.shape[1]):
            # spec[:, i] = (mu * np.abs(spec[:, i])) / (1+np.log( mu))
            spec[:, i] = ((np.sign(spec[:, i]) * np.log(1 + mu * np.abs(spec[:, i])))) / (1 + np.log(mu))

    kernel=np.hanning(sm_win)
    for i in range(spec.shape[1]):
        spec[:, i] = np.convolve(spec[:, i],kernel,'same')

    if test:
        spec=np.gradient(spec, axis=0)
        spec= np.clip(spec, 0, None, out=spec)
        for i in range(spec.shape[0]):
            spec[i,:]=np.mean(spec[i,:])

    return spec



# superlux from madmom (Boeck et al)
def superflux(spec_x=[], A=None, B=None, win_size=8):
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
        kernel = np.hamming(win_size)

        B = np.convolve(B, kernel, 'same')
        # B = B / max(B)
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
    return np.rint(np.asanyarray(samples) / (hop_length / hops_per_frame))


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
        # print('fscore: ',e)
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


def extract_tempo(onsets=None, window_size_in_s=8, constant_tempo=True):
    LH = HOP_SIZE
    win_len_s = window_size_in_s
    N = int(SAMPLE_RATE / LH * win_len_s)
    min_bpm = int(DEFAULT_TEMPO / 4)  # 30
    max_bpm = int(DEFAULT_TEMPO * 2)  # 240
    bpm_slice = max_bpm - min_bpm
    tic = time.clock()
    # N=bpm_slice

    # n_original = onsets.shape[0]
    # n_power_of_2 = 2 ** int(np.ceil(np.log2(n_original)))
    # n_pad = n_power_of_2 - n_original
    # Pad onsets and perform librosa Autocorrelation tempogram
    onsets = np.pad(onsets, int(N // 2) + 1,
                    mode='reflect')
    n_frames = 1 + int((len(onsets) - N))
    fonsets = np.lib.stride_tricks.as_strided(onsets, shape=(N, n_frames),
                                              strides=(onsets.itemsize, onsets.itemsize))

    powerspec = np.abs(fft.fft(fonsets, axis=0)) ** 2

    autocorr = fft.ifft(powerspec, axis=0, overwrite_x=True)  # [:int(powerspec.shape[0] / 2)]
    sff = (autocorr.real / autocorr.real.max())[0:N, :]

    # mean calcs.
    sff_mean = np.mean(sff, axis=1, keepdims=True)
    # perform librosa tempo extraction with two iterations
    bpms = librosa.core.tempo_frequencies(sff.shape[0], hop_length=LH, sr=SAMPLE_RATE)

    # showFFT(sff,bpms)
    prior_mean = np.exp(-0.5 * ((np.log2(bpms) - np.log2(DEFAULT_TEMPO)) / 1.) ** 2)
    best_period_mean = np.argmax(sff_mean * prior_mean[:, np.newaxis], axis=0)

    # first find the most common tempo of the take
    tempi_mean = bpms[best_period_mean]


    # then force the tempi to that area
    prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(tempi_mean)) / 0.2) ** 2)
    best_period = np.argmax(sff * prior[:, np.newaxis], axis=0)
    tempi = bpms[best_period]

    # Wherever the best tempo is index 0, return start_bpm
    tempi[best_period == 0] = min_bpm
    #showEnvelope(tempi)
    #Get constant tempo estimate
    if constant_tempo:
        tempi[:]=tempi_mean
        tempi_smooth=tempi
    else:
        kernel=np.hanning(int(N/1.5))
        tempi_pad=np.array(list(tempi[:200])+list(tempi)+list(tempi[-200:]))
        tempi_smooth=np.convolve(tempi_pad/kernel.sum(),kernel, 'same')[200:-200]
        #showEnvelope(tempi_smooth)
    # Tempo and it's manifolds
    d = DEFAULT_TEMPO
    targets = [d / 4, d / 2, d, d * 2, d * 4]
    target_tempos=[]
    # Change tempi to tempo quantization multipliers
    for i in range(tempi.size):
        target_tempos.append(min(targets, key=lambda x: abs(x - tempi_smooth[i])))
        #This results in bad behaviour when the tempo shifts over mean of two targets DO NOT USE
        #tempo= min(targets, key=lambda x: abs(x - tempi_smooth[i]))
        #tempi_smooth[i] = tempi_smooth[i] / tempo
    target_median=np.median(target_tempos)
    tempi_smooth[:] = tempi_smooth[:] / target_median
    print('\ntempomap time:{}'.format(time.clock() - tic))
    return (tempi_smooth)


def conform_time(X, tempomap, quantize=False, round=1):
    """
    Conforms the hits X according to a tempomap
    :param X: numpy array, The onsets to quantize
    :param tempomap: numpy array, the tempo modifiers for each frame
    :param quantize: boolean, quantize the conformed X to grid of 16th notes at 4 times the default bpm
    :return: numpy array, quantized onsets

    Notes:
    """
    # Shortest allowed note in seconds 16th note at 480bpm
    if not len(X):
        return []
    shortest_note = DEFAULT_TEMPO / 60 / SXTH_DIV / 4
    # return value space
    retX = np.zeros((X.size))
    # gap of beginning to the first hit as time value
    X = X.astype(int)

    newgap = frame_to_time(sum(tempomap[:X[0]]))
    # newgap = np.rint(newgap / shortest_note) * shortest_note
    # store first hit
    retX[0] = newgap
    # retX[0] = np.rint(retX[0] / shortest_note) * shortest_note
    # iterate over all hits
    for i in range(1, X.size):
        # Calculate the gap between two consecutive hits
        newgap = frame_to_time(sum(tempomap[X[i - 1]:X[i]]))
        # newgap = np.rint(newgap / shortest_note) * shortest_note
        # move the hit to last hit+newgap
        retX[i] = retX[i - 1] + newgap
        if False:
            retX[i] = np.rint(retX[i] / shortest_note) * shortest_note
    # if hits are to be quantized
    if True:
        # iterate over hits
        for i in range(retX.size):
            # Move return value to closest note measure by truncating the float decimal
            retX[i] = np.rint(retX[i] / shortest_note) * shortest_note

    # return frame values of conformed X But use resolution of shortest_note
    # 88.2ms.
    return time_to_frame(retX, hop_length=Q_HOP)


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
    # threshold=thresh*np.median(localMaxima)+delta
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




def NMFD(X, iters=500, Wpre=[], include_priors=False):
    """
    :param :
       - X : The spectrogram
       - iters : # of iterations
       - Wpre : Prior vectors of W

    :return :
       - H : Activations of different drums in X

    """
    # epsilon, added to matrix elements to ensure non-negativity in matrices
    eps = 10 ** -18

    originalN = X.shape[1]
    # Add samples to the audio to ensure all levels are accounted for
    if include_priors:
        for i in range(Wpre.shape[1]):
            pass
            X = np.hstack((X, .5*Wpre[:, i, :]))

    #add noise priors, do not use, minimal improvement with serious speed drop.
    #W_z=np.random.rand(Wpre.shape[0],Wpre.shape[1], Wpre.shape[2])
    #Wpre=np.concatenate((Wpre, W_z), axis=1)

    # data size
    M, N = X.shape
    W = Wpre
    M, R, T = W.shape

    # Initial H, non negative, non zero. any non zero value works
    H = np.full((R, N), .15)

    # Make sure the input is normalized
    W = W / W.max()
    X = X / X.max()

    # all ones matrix
    repMat = np.ones((M, N))

    # Spaceholder for errors
    err = np.zeros(iters)

    # KLDivergence for cost calculation
    def KLDiv(x, y, LambHat):
        return (x * np.log(LambHat) - x + y).sum()

    def ISDiv(x):
        return (x - np.log(x) - 1).sum()

    def sumLambda(W, H):
        Lambda = np.zeros((W.shape[0], H.shape[1]))
        shifter = np.zeros((R, N + T + 1))
        shifter[:, T:-1] = H
        for t in range(T):
            Lambda += W[:, :, t] @ shifter[:, T - t: -(t + 1)]
        return Lambda

    mu = 0.38  # sparsity constraint
    for i in range(iters):
        if REQUEST_RESULT:
            return H
        # print('\rNMFD iter: %d' % (i + 1), end="", flush=True)

        if i == 0:
            Lambda = eps + sumLambda(W, H)
            LambHat = X * Lambda ** -1 + eps

        shifter = np.zeros((M, N + T))
        shifter[:, :-T] = LambHat
        Hhat1 = np.zeros((R, N))
        Hhat2 = np.zeros((R, N))
        for t in range(T):
            Wt = W[:, :, t].T
            Hhat1 += Wt @ shifter[:, t: t + N]
            Hhat2 += Wt @ repMat + eps
        H = H * Hhat1 / (Hhat2 + mu)

        # precompute for error and next round
        Lambda = eps + sumLambda(W, H)
        LambHat = X * (Lambda ** -1) + eps

        # 2*eps to ensure the eps don't cancel out each other in the division.
        # err[i] = KLDiv(X + eps, Lambda + 2 * (eps), LambHat + eps)
        err[i] = ISDiv(LambHat + eps)

        if (i >= 1):
            errDiff = (abs(err[i] - err[i - 1]) / (err[0] - err[i] + eps))
            # print(errDiff)
            if errDiff < 0.0003:# or err[i] > err[i - 1]:
                break

        #Adaptive W - no considerable improvement
        if (False):
            What1 = np.zeros((M,R, T))
            What2 = np.zeros((M,R, T))
            shifter = np.zeros((N+T, R))
            shifter[:-T, :] = H.T[:,:]
            for t in range(T):
                W[:, :, t] = W[:, :, t] * (LambHat@shifter[t:-(T-t), :] / (repMat@shifter[t:-(T-t),:] + eps+(mu/T)))
            #dittmar &al. alpha
            #alpha=(i/iters)**4
            #fix Wpre: alpha=1, adapt freely: alpha=0
            alpha=0.1
            W=Wpre*alpha+(1-alpha)*W
            W = W / W.max()
    return H,W, err / err.max()

#
#
# import matplotlib.pyplot as plt
# import matplotlib.cm as cmaps
#
#
# def showEnvelope(env, legend=None, labels=None):
#     """
#     Plots an envelope
#     i.e. an onset envelope or some other 2d data
#     :param env: the data to plot
#     :return: None
#     """
#     if 0 > 1:
#
#         f, axarr = plt.subplots(len(env),1 , sharex=True, sharey=True)
#         for i in range(len(env)):
#             axarr[i].plot(env[i][0], label='Onset envelope')
#             axarr[i].vlines(env[i][1], 0, 1, color='r', alpha=0.9,linestyle='--', label='Onsets')
#             axarr[i].get_xaxis().set_visible(False)
#             axarr[i].get_yaxis().set_ticks([])
#         f.subplots_adjust(hspace=0.1)
#         axarr[0].set(ylabel='Kick, superflux')
#         axarr[1].set(ylabel='Kick, activations H')
#         #axarr[2].set(ylabel='Closed Hi-Hat')
#         axarr[len(env)-1].get_xaxis().set_visible(True)
#         plt.savefig("Onset_env_peaks_alg1.png")
#         plt.tight_layout()
#     else:
#         plt.figure(figsize=(10, 2))
#         plt.plot(env)
#         plt.xlim(xmin=1)
#         plt.show()
#         #ax = plt.subplot(111)
#         if legend != None:
#             box = ax.get_position()
#             ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#             plt.gca().legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
#             # plt.gca().legend(legend, loc='right')
#         if labels != None:
#             # my_xticks=[str(x)[:4] for x in np.linspace(0.2,0.4,int(env.shape[0]/2))]
#             #my_xticks = np.arange(0, env.shape[0], .01)
#             #plt.xticks(np.arange(1, env.shape[0], 1), range(1,20))
#
#             plt.xlabel(labels[0], fontsize=12)
#             plt.ylabel(labels[1], fontsize=12)
#
#         #plt.savefig("nadam_lr.png")
#         #plt.tight_layout()

#
# def showFFT(env, ticks=None):
#     """
#     Plots a spectrogram
#
#     :param env: the data to plot
#     :return: None
#     """
#     if len(env) > 1:
#
#         f, axarr = plt.subplots(1, len(env), sharex=True, sharey=True)
#
#         for i in range(len(env)):
#
#             axarr[i].imshow(env[i], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
#
#             #axarr[i].get_xaxis().set_visible(False)
#         # axarr[1].imshow(env[1], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
#         # axarr[1].set(xlabel='Snare frames')
#         # axarr[2].imshow(env[2], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
#         # axarr[2].set(xlabel='Closed Hi-Hat frames')
#         f.subplots_adjust(wspace=0.03)
#         axarr[0].set(ylabel='STFT bin')
#         f.text(0.5, 0.02, 'k', ha='center', va='center')
#
#         plt.savefig("Templates3.png")
#
#     if ticks != None:
#         top = len(ticks)
#         plt.imshow(np.flipud(env), aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
#         my_yticks = ticks
#         plt.xlabel('frame', fontsize=12)
#         plt.ylabel('tempo', fontsize=12)
#         plt.yticks(np.arange(0, top, 10), np.rint(np.fliplr([ticks[0:top:10], ]))[0])
#     plt.tight_layout()
#
#     # plt.figure(figsize=(10, 4))
#     # plt.xlabel('frame', fontsize=12)
#     # plt.ylabel('stft bin', fontsize=12)
#     # # f, axarr = plt.subplots(3, sharex=True, sharey=True)
#     # plt.subplot(131, sharex=True, sharey=True)
#     # plt.imshow(env[0], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
#     # plt.subplot(132)
#     # plt.imshow(env[1], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
#     # plt.subplot(133)
#     # plt.imshow(env[2], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))


def acceptHit(value, hits):
    """
    Helper method to clear mistakes of the annotation such as ride crash and an open hi hat.
    :param value: int the hit were encoding
    :param hits: binary string, The hits already found in that location
    :return: boolean, if it is ok to annotate this drum here.
    """
    # Tähän pitää tehä se et vaan vaikka 3 iskua yhteen frameen!
    # Pitää vielä testata

    sum = 0
    for i in range(value):
        if np.bitwise_and(i, hits):
            sum += 1
        if sum > 3:
            return False
    ##Voiko tän jättää pois??
    ##Ainakin tekee kökköä kamaa, mielummin ilman
    offendingHits = np.array([
        # value == 2 and np.bitwise_and(3, hits),
        value == 3 and np.bitwise_and(8, hits),
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
        pass
        # return False
    return True


def truncZeros(frames):
    zeros = 0
    for i in range((frames.size)):
        if frames[i] == 0:
            zeros += 1
            # longest pause
            if zeros == 32:
                frames[i - zeros + 1] = -zeros
                zeros = 0
                continue
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
    sixtDivider = SAMPLE_RATE / Q_HOP / SXTH_DIV
    if QUANTIZE:
        for i in range(len(a)):
            # Move frames to nearest sixteenth note
            a[i] = (np.rint(a[i][0] / sixtDivider),) + a[i][1:]
    # Define max frames from the first detected hit to end
    print(len(a[-1]))
    if (len(a) == 0):
        return []
    maxFrame = int(a[-1][0] - a[0][0] + 1)
    # print(maxFrame)
    # spaceholder for return array
    frames = np.zeros(maxFrame, dtype=int)
    for i in range(len(a)):
        # define true index by substracting the leading empty frames
        index = int(a[i][0] - a[0][0])
        if index>=frames.shape[0]:
            break
        # The actual hit information
        value = int(a[i][1])
        # Encode the hit into a charachter array, place 1 on the index of the drum #
        if acceptHit(value, frames[index]):
            try:
                frames[index] = np.bitwise_or(frames[index], 2 ** value)
            except:
                print (frames[index],value)

    # return array of merged hits starting from the first occurring hit event
    if ENCODE_PAUSE:
        frames = truncZeros(frames)
    print(len(frames))
    return frames


def splitrowsanddecode(a, deltaTempo=1.0):
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
    print(deltaTempo)
    frameMul = 1/deltaTempo
    if False:
        frameMul = SAMPLE_RATE / Q_HOP / SXTH_DIV
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
                decodedFrames.append([int((i + pause) * frameMul), abs(j - (nrOfDrums - 1))])
        i += 1
    # return the split hits
    return decodedFrames


def dec_to_binary(f):
    """
    Returns a binary representation on a given integer
    :param f: an integer
    :return: A binary array representation of f
    """
    return format(f, "0{}b".format(nrOfDrums))


#########################################################
#                                                       #
#   The code below was used in research only, not in    #
#   the final game demo. Feel free to read trough it    #
#   but do not try to use it as is. Nothing below this  #
#   disclaimer is tested and safe to use!               #
#                                                       #
#########################################################


def semi_adaptive_NMFB(X, Wpre, iters=100):
    epsilon = 10 ** -18
    X = X + epsilon
    # Use only the first frame of NMFD prior
    W = Wpre[:, :, 0]
    # Initial H, non negative, non zero. any non zero value works
    H = np.full((W.shape[1], X.shape[1]), .5)
    # normalize input
    W = W / W.max()
    X = X / X.max()
    # error space
    err = np.zeros(iters)
    Wt = W.T

    def ISDiv(x, y):
        return ((x / y + epsilon) - np.log(x / y + epsilon) - 1).sum()

    WH = epsilon + np.dot(W, H)
    mu = 0.38
    for i in range(iters):
        Hu = (Wt @ (X * WH ** (- 2)))
        Hd = (epsilon + (Wt @ WH ** (- 1)))
        H = H * Hu / (Hd + mu)
        WH = epsilon + np.dot(W, H)
        err[i] = ISDiv(X + epsilon, WH + epsilon)  # + (sparsity * normH)
        if (i >= 1):
            errDiff = (abs(err[i] - err[i - 1]) / (err[1] - err[i] + epsilon))
            if errDiff < 0.0003 or err[i] > err[i - 1]:
                pass
                break
    return H, err / err.max()


def HFC_filter(X=[]):
    print(X.shape)
    for i in range(X.shape[1]):
        X[:, i] = X[:, i] / X[:, i].max()
    # X=X/X.max()
    return (X)


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
        i = int(i)
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


def getTempomap(H):
    if H.ndim < 2:
        onsEnv = H
    else:
        onsEnv = np.sum(H, axis=0)
    bdTempo = (librosa.beat.tempo(onset_envelope=onsEnv, sr=SAMPLE_RATE, hop_length=HOP_SIZE,
                                  ac_size=2))  # aggregate=None))
    # bdAvg = movingAverage(bdTempo, window=30000)
    # avgTempo = np.mean(bdAvg)
    # return tempoMask(bdAvg), avgTempo
    return bdTempo


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
    frameLength = SAMPLE_RATE / HOP_SIZE
    # extract indices in X that correspond to a 16th note in tempo at time t
    # by calculating modulo and taking the inverse boolean array mod(a,b)==0 ==True
    indices = np.array([int((s % (sixtLength[s] / frameLength))) for s in range(0, tempos.shape[0])])
    invertedIndices = np.invert(indices.astype('bool'))
    return invertedIndices


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
    for i in range(Y.shape[1]):
        Y[:, i] = np.sign(Y[:, i]) * np.log(1 + mu * Y[:, i]) / np.log(1 + mu)
    # for n in range(Y.shape[0]):
    #    for i in range(Y.shape[1]):
    #        # x_i_n=Y[n,i].flatten()@Y[n,i].flatten()
    #        x_i_n = Y[n, i]
    #        x_mu[n, i] = np.sign(Y[n, i]) * np.log(1 + mu * x_i_n) / np.log(1 + mu)
    return Y


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


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
    T = (np.arange(0, (win_len), 1) / sr).T

    win_num = int((novelty_curve.size - noverlap) / (win_len - noverlap))
    # print(novelty_curve.shape,T.shape,win_num, len(omega))
    x = np.zeros((win_num, len(omega)))
    t = np.arange(int(win_len / 2), int(novelty_curve.size - win_len / 2), hopsize) / sr

    tpiT = 2 * np.pi * T

    # for each frequency given in f
    for i in range(omega.size):
        tpift = omega[i] * tpiT
        cosine = np.cos(tpift)
        sine = np.sin(tpift)

        for w in range(win_num):
            start = (w) * hopsize
            stop = start + win_len
            sig = novelty_curve[start:stop] * win
            co = sum(sig * cosine)
            si = sum(sig * sine)
            x[w, i] = (co + 1j * si)

    return t, x.T


def squeeze(sff, tempo_target):
    sff2 = np.zeros((sff.shape))
    for i in range(sff.shape[1]):
        j = max_bpm - min_bpm - 1
        while j > tempo_target * 2:
            sff2[int(j / 2), i] += sff[j, i]
            j -= 1
        j = 0
        while j < tempo_target / 2:
            sff2[int(j * 2), i] += sff[j, i]
            j += 1
    return sff2


def get_cmask(tempo, tempomask):
    pass

###From extract_tempo
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

# onsets=onsets[:-99]-running_mean(onsets,100)
# onsets = np.array([0 if i < 0 else i for i in onsets])
# onsets=onsets/onsets.max()

###From pick_onsets
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
from sklearn.decomposition import FastICA

def PSA(X,fpr):
    eps = 10 ** -18
    X=X.T
    #fpr = fpr[:, :, 0]
    print (fpr.shape)
    fpr=fpr/fpr.max()
    fpp=np.linalg.pinv(fpr)
    print(fpp.shape)
    tHat = (fpp@X).T
    #for i in range(9):
    #    showEnvelope(tHat.T[i])
    print(tHat.shape)
    ica = FastICA(n_components=9)
    t = (ica.fit_transform(tHat))
    print(t.shape)
    tp=np.linalg.pinv(t)
    f=X@tp.T
    print(f.shape)
    f=f/f.max()
    def KLDiv(x, y):
        return (x * (np.log(y/x+eps) - x + y)).sum()
    def ISDiv(x,y):
        return (y/x - np.log(y/x) - 1).sum()
    retf=np.zeros_like(f)
    print(fpr.shape, f.shape)
    for i in range(f.shape[1]):
        x=fpr.T
        y=f.T

        y=abs(y)
        mini=ISDiv(x[i], y[0])
        #print(x[i], y[0])
        index=0
        for j in range(y.shape[0]-1):
            if mini>=ISDiv(x[i], y[j]):
                mini=ISDiv(x[i], y[j])
                index=j
        print (index, mini)
        retf.T[i]=y[index]
    tTilde = (np.linalg.pinv(f) @ X).T
    tTilde = np.clip(np.abs(tTilde.T), 0, None)
    showEnvelope(tTilde.T)
    for i in range(9):
        showEnvelope(tTilde[i]/tTilde[i].max())
    return t, f