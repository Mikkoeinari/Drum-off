#from time import time
import time
import madmom

#Uncomment to plot, conflicts with kivy!
#import matplotlib.cm as cmaps
#import matplotlib.pyplot as plt
import OPTICS
import numpy as np
import scipy
from scipy import fftpack as fft
from scipy.ndimage.filters import median_filter, maximum_filter
from scipy.signal import argrelmax
from sklearn.mixture import BayesianGaussianMixture

FRAME_SIZE = 2 ** 11
HOP_SIZE = 2 ** 9
# After quantization to 120bpm we could use hop size 1378.125 the resolution of 64th note at 120bpm
# or 918.75 the 64th triplet with every third frame the 32th.
# Float presicion is allowed since the hop size is only used for time to frame calculations.

Q_HOP = 689.0625*2 #have options[459.375,612.5,689.0625,1378.125]*[.5,1,2,4,8,16]
SAMPLE_RATE = 44100
FREQUENCY_PRE = np.ones((24))  # [0,16384]#0-2^14
MIDINOTE = 36  # kickdrum in standard midi notation
THRESHOLD = 0.0
PROBABILITY_THRESHOLD = 0.0
DEFAULT_TEMPO = 120  # 175.44bpm 1/16 with hop 516
DRUMKIT_PATH = '../trainSamplet/'
REQUEST_RESULT = False
DELTA = 0.15
midinotes = [36, 38, 42, 46, 44, 50, 48, 47, 43, 41, 51, 49, 57,
             55]  # BD, SN, CHH, OHH,SHH, TT,TT2,TT3, FT,FT2, RD, CR,CR2,splash, Here we need generality
nrOfDrums = 14  # Maximum kit size
nrOfPeaks = 16  # IF CHANGED ALL PREVIOUS SOUNDCHECKS INVALIDATE!!!
max_n_frames = 10
total_priors = 0
MS_IN_MIN = 60000
SXTH_DIV = 16
QUANTIZE = False
ENCODE_PAUSE = True
_ImRunning = False
# Bark scale filterbank Shortest processing time good results
proc = madmom.audio.filters.BarkFilterbank(
    madmom.audio.stft.fft_frequencies(num_fft_bins=int(FRAME_SIZE / 2), sample_rate=SAMPLE_RATE),
    num_bands='double', fmin=20.0, fmax=15500.0, norm_filters=False, unique_filters=True)
# Mel scale filterbank
proc2 = madmom.audio.filters.MelFilterbank(
    madmom.audio.stft.fft_frequencies(num_fft_bins=int(1024 / 2), sample_rate=SAMPLE_RATE),
    num_bands=20, fmin=20.0, fmax=17000.0, norm_filters=False, unique_filters=True)


# Logarithmic filterbank
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


def get_possible_notes(drum_kit=None):
    """
    Defines all notes producable with a human body and given drum kit
    :param drum_kit: List of integers, drum identifiers of a drum kit
    For instance kick, snare and closed hi-hat is [0,1,2]
    :return: numpy array of integers, notes playable on the kit plus pause notes.
    """
    from itertools import combinations
    # Pause notes[-1,...,-32]
    pauses = [-i for i in range(1, 32)]
    # remove kick from combinations, add back later
    kickless = drum_kit[1:]
    # Possible hits with 2 hands
    hits_without_kick = list(combinations(kickless, 2))
    # Add possible hits with one hand
    hits_without_kick = hits_without_kick + [(i,) for i in kickless]
    # Add kick to possible hits
    hits_with_kick = [h + (drum_kit[0],) for h in hits_without_kick]
    # concatenate all and add a solo kick drum
    hits_with_and_without_kick = hits_with_kick + hits_without_kick + [(drum_kit[0],)]
    # Notes array
    possible_notes = np.zeros(len(hits_with_and_without_kick)).astype(int)
    # Encode notes
    for i in range(len(hits_with_and_without_kick)):
        for j in hits_with_and_without_kick[i]:
            possible_notes[i] = np.bitwise_or(possible_notes[i], 2 ** int(j))
    # Add pauses to the notes
    possible_notes = np.concatenate((possible_notes, pauses))
    # return possible notes
    return (possible_notes)

def to_midinote(notes):
    return list(midinotes[i] for i in notes)


def findDefBinsBG(frames, filteredSpec, ConvFrames, K, bs_len=32):
    """
    Calculate the prior vectors for W to use in NMF
    :param frames: Numpy array of hit locations (frame numbers)
    :param filteredSpec: Spectrogram, the spectrogram where the vectors are extracted from
    :param ConvFrames: int, number of frames the priors contain
    :param K: int, max number of priors per drum
    :param bs_len: int, Not used
    :return: tuple of Numpy arrays, prior vectors Wpre,heads for actual hits and tails for decay part of the sound
    """

    global total_priors, max_n_frames
    max_n_frames=ConvFrames
    gaps = np.zeros((frames.shape[0], max_n_frames))
    # gaps = np.zeros((frames.shape[0], ConvFrames))
    for i in range(frames.shape[0]):
        for j in range(ConvFrames):
            gaps[i, j] = frames[i] + j

    a = np.reshape(filteredSpec[gaps.astype(int)], (nrOfPeaks, -1))
    a2 = a
    # a2=resample(a, n_samples=bs_len, replace=True, random_state=2)
    bgMeans = BayesianGaussianMixture(
        n_components=K, covariance_type='spherical',  # weight_concentration_prior=1,
        weight_concentration_prior_type='dirichlet_process',
        mean_precision_prior=.8,  # covariance_prior=np.eye(proc.shape[1]*ConvFrames),
        init_params="random", max_iter=100, random_state=2).fit(a2)
    K1 = np.unique(bgMeans.predict(a2))
    # print(K1)
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
    a2 = a
    # a2 = resample(a, n_samples=bs_len, replace=True, random_state=2)
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


def getStompTemplate():
    """
    records sound check takes
    :return: numpy array, the recorded audio
    """
    global _ImRunning

    _ImRunning = True

    buffer = np.zeros(shape=(2646000))
    j = 0
    time.sleep(0.1)
    strm = madmom.audio.signal.Stream(sample_rate=SAMPLE_RATE, num_channels=1, frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    for i in strm:

        buffer[j:j + HOP_SIZE] = i[:HOP_SIZE]
        j += HOP_SIZE
        if j >= 2646000 or (not _ImRunning):
            buffer[j:j + 6000] = np.zeros(6000)
            strm.close()
            return buffer[:j + 6000]


def getPeaksFromBuffer(filt_spec, numHits):
    """

    :param filt_spec: numpy array, the filtered spectrogram containing sound checked drum audio

    :param numHits: int, the number of hits to recover from filt_spec

    :return: numpy array, peak locations in filt_spec
    """
    threshold = 1
    searchSpeed = .1
    # peaks=cleanDoubleStrokes(madmom.features.onsets.peak_picking(superflux_3,threshold),resolution)
    H0 = superflux(spec_x=filt_spec.T, win_size=8)

    # peaks = cleanDoubleStrokes(pick_onsets(H0 / H0.max(), delta=threshold), resolution)
    peaks = pick_onsets(H0, threshold=threshold)
    changed = False
    last = 0
    while (peaks.shape != (numHits,)):
        # Make sure we don't go over numHits
        # There is a chance of an infinite loop here!!! Make sure that don't happen
        if (peaks.shape[0] > numHits) or (peaks.shape[0] < last):
            if changed == False:
                searchSpeed = searchSpeed / 2
            changed = True
            threshold += searchSpeed
        else:
            changed = False
            threshold -= searchSpeed
        # peaks=cleanDoubleStrokes(madmom.features.onsets.peak_picking(superflux_3,threshold),resolution)
        # peaks = cleanDoubleStrokes(pick_onsets(H0, delta=threshold), resolution)
        last = peaks.shape[0]
        peaks = pick_onsets(H0, threshold=threshold)
    return peaks


def recalculate_thresholds(filt_spec, shifts, drums, drumwise=False, method='NMFD'):
    """

    :param filt_spec: numpy array, The spectrum containing sound check
    :param shifts: list of integers, the locations of different drums in filt_spec
    :param drums: list of drums
    :param drumwise: boolean, if True the thresholds will be calculated per drum,
     if False a single trseshold is used for the whole drumkit
    :param rm_win: int, window size.
    :return: None
    """
    onset_alg = 2
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
    if method=='NMFD':
        H, Wpre, err1 = NMFD(filt_spec.T, iters=128, Wpre=Wpre, include_priors=True, n_heads=total_heads)
    else:
        H, err1=semi_adaptive_NMFB(filt_spec.T,  Wpre=Wpre,iters=128,n_heads=total_heads)
    total_heads = 0
    Hs = []
    for i in range(len(drums)):
        heads = drums[i].get_heads()
        K1 = heads.shape[2]
        ind = total_heads
        if onset_alg == 0:
            for k in range(K1):
                index = ind + k
                HN = superflux(A=sum(Wpre.T[0, index, :]), B=H[index], win_size=3)
                #HN = energyDifference(H[index], win_size=6)
                # HN = HN / HN.max()
                if k == 0:
                    H0 = HN
                else:
                    H0 = np.maximum(H0, HN)
                total_heads += 1
        elif onset_alg == 1:
            for k in range(K1):
                index = ind + k
                HN = get_preprocessed_spectrogram(A=sum(Wpre.T[0, index, :]), B=H[index], test=True)[:, 0]
                if k == 0:
                    H0 = HN
                else:
                    H0 = np.maximum(H0, HN)
                total_heads += 1
                H0 = H0 / H0.max()
        else:
            kernel = np.hanning(6)
            for k in range(K1):
                index = ind + k
                HN = H[index]
                # HN = np.convolve(HN, kernel, 'same')
                HN = HN / HN.max()
                if k == 0:
                    H0 = HN
                else:
                    # H0 = np.maximum(H0, HN)
                    H0 += HN
                total_heads += 1
            # H0 = H0 / H0.max()
        # H0 = H0[:-(rm_win - 1)] - running_mean(H0, rm_win)
        # H0 = np.array([0 if i < 0 else i for i in H0])
        H0 = H0 / H0.max()
        Hs.append(H0)

        besthits = []
        if drumwise:
            deltas = np.linspace(0., 1, 100)
            f_zero = 0
            threshold = 0
            maxd = 0
            for d in deltas:
                if i < len(shifts) - 1:
                    peaks = pick_onsets(H0[shifts[i]:shifts[i + 1]], threshold=d)
                    # peaks=madmom.features.onsets.peak_picking(H0[shifts[i]:shifts[i+1]], d)
                    # showEnvelope(H0[shifts[i]:shifts[i+1]])
                else:
                    peaks = pick_onsets(H0[shifts[i]:], threshold=d)
                # print(peaks.shape[0])

                drums[i].set_hits(peaks[np.where(peaks < filt_spec.shape[0] - 1)])
                predHits = drums[i].get_hits()
                actHits = drums[i].get_peaks()
                trueHits = k_in_n(actHits, predHits, window=1)
                prec, rec, f_drum = f_score(trueHits, predHits.shape[0], actHits.shape[0])
                # print(d,f_drum)
                if f_drum == f_zero:
                    maxd = d
                if f_drum > f_zero:
                    f_zero = f_drum
                    threshold = d
            # if optimal threshold is within a range [threshold, maxd] find a sweet spot empirically,
            #  increasing alpha lowers threshold and decreases precision
            if maxd > 0:
                alpha = 0.55
                beta = 1 - alpha
                threshold = (alpha * threshold + beta * maxd)

            # arbitrary minimum threshold check
            threshold = max((threshold, 0.05))

            print('delta:', threshold, f_zero)
            drums[i].set_threshold(threshold)
            # drums[i].set_threshold(.2)

    if not drumwise:
        # print(len(Hs))
        deltas = np.linspace(0., 1, 100)
        f_zero = 0
        threshold = 0
        maxd = 0
        for d in deltas:
            precision, recall, fscore, true_tot = 0, 0, 0, 0
            for i in range(len(drums)):
                peaks = pick_onsets(Hs[i], threshold=d)
                drums[i].set_hits(peaks[np.where(peaks < filt_spec.shape[0] - 1)])
                predHits = drums[i].get_hits()
                # print(predHits)
                actHits = drums[i].get_peaks() + shifts[i] - 1
                # print(actHits)
                trueHits = k_in_n(actHits, predHits, window=3)
                prec, rec, f_drum = f_score(trueHits, predHits.shape[0], actHits.shape[0])
                precision += prec * actHits.shape[0]
                recall += rec * actHits.shape[0]
                fscore += (f_drum * actHits.shape[0])
                true_tot += actHits.shape[0]
            # if (fscore / true_tot) > f_zero:
            #     #print(fscore/ true_tot)
            #     f_zero = (fscore / true_tot)
            #     threshold = d
            if (fscore / true_tot) == f_zero:
                maxd = d
            if (fscore / true_tot) > f_zero:
                f_zero = (fscore / true_tot)
                threshold = d

        # if optimal threshold range, increase a bit
        if maxd > 0:
            alpha = 0.666
            beta = 1 - alpha
            threshold = (alpha * threshold + beta * maxd)

        # arbitrary minimum threshold check
        threshold = max((threshold, 0.15))
        # print('delta:', threshold, f_zero)
        for i in range(len(drums)):
            drums[i].set_threshold(threshold)

            # constant delta
            # drums[i].set_threshold(0.16)

            # Try loosening plate drum thresholds: NOT FINAL!!!
            if i in [2, 3, 6, 7, 8]:
                pass
                # drums[i].set_threshold(threshold)

    # mean = sum(drums[i].get_threshold() for i in range(len(drums))) / len(drums)
    # for i in range(len(drums)):
    #    drums[i].set_threshold(mean)


def liveTake():
    """
    records a drum take
    :return:
    """
    global _ImRunning
    _ImRunning = True
    buffer = np.zeros(shape=(44100 * 10 + 18000))  # max take length, must make user definable

    j = 0
    time.sleep(0.1)
    strm = madmom.audio.signal.Stream(sample_rate=SAMPLE_RATE, num_channels=1, frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    for i in strm:
        buffer[j:j + HOP_SIZE] = i[:HOP_SIZE]
        j += HOP_SIZE
        if j >= buffer.shape[0] - 18000 or (not _ImRunning):
            buffer[j:j + 6000] = np.zeros(6000)
            strm.close()
            # Should this yield instead of returning? To record as long as the drummer wants...
            return buffer[:j + 6000]


def processLiveAudio(liveBuffer=None, drums=None, quant_factor=1.0, iters=0, method='NMFD', thresholdAdj=0.):
    """
    main logic for source separation, onset detection and tempo extraction and quantization
    :param liveBuffer: numpy array, the source audio
    :param drums: list of drums
    :param quant_factor: float, amount of quantization (change to boolean)
    :param iters: int, number of runs of nmfd for bagging separation
    :param method: The source separation method, 'NMF' or 'NMFD
    :param thresholdAdj: float, adjust the onset detection thresholds, one value for all drums.
    :return: list of drums containing onset locations in hits field and mean tempo of the take
    """

    onset_alg = 2
    filt_spec = get_preprocessed_spectrogram(liveBuffer, sm_win=4)
    stacks = 1
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
    for i in range(int(stacks)):
        if method == 'NMFD' or method == 'ALL':
            H, Wpre, err1 = NMFD(filt_spec.T, iters=iters, Wpre=Wpre, include_priors=True, n_heads=total_heads, hand_break=True)
        if method == 'NMF' or method == 'ALL':
            H, err2 = semi_adaptive_NMFB(filt_spec.T, Wpre=Wpre, iters=iters, n_heads=total_heads, hand_break=True)
        if method == 'ALL':
            errors = np.zeros((err1.size, 2))
            errors[:, 0] = err1
            errors[:, 1] = err2

            # showEnvelope(errors, ('NMFD Error', 'NMF Error'), ('iterations', 'error'))
        if i == 0:
            WTot, HTot = Wpre, H
        else:
            WTot += Wpre
            HTot += H
    Wpre = (WTot) / stacks
    H = (HTot) / stacks

    onsets = np.zeros(H[0].shape[0])
    total_heads = 0
    picContent = []
    allPeaks = []
    # showEnvelope(superflux(A=sum(Wpre.T[:, 2, :]), B=H[2],win_size=2)[:500])
    # showEnvelope(energyDifference(H[2], win_size=2)[:500])
    # showEnvelope([(H[1]/H[1].max())[100:600], 0.09090909090909091,0.2318181818181818, 0.4040404040404041])
    times = 0
    for i in range(len(drums)):
        # if i<=9:
        #    showEnvelope(H[i][:1500])
        heads = drums[i].get_heads()
        K1 = heads.shape[2]
        ind = total_heads

        if onset_alg == 0:
            for k in range(K1):
                index = ind + k
                HN = superflux(A=sum(Wpre.T[0, index, :]), B=H[index],win_size=3)
                #HN = energyDifference(H[index], win_size=6)
                # HN = HN / HN.max()
                if k == 0:
                    H0 = HN
                else:
                    H0 = np.maximum(H0, HN)
                total_heads += 1
        elif onset_alg == 1:
            for k in range(K1):
                index = ind + k
                HN = get_preprocessed_spectrogram(A=sum(Wpre.T[0, index, :]), B=H[index], test=True)[:, 0]
                if k == 0:
                    H0 = HN
                else:
                    H0 = np.maximum(H0, HN)
                total_heads += 1
                H0 = H0 / H0.max()
        else:
            kernel = np.hanning(6)
            for k in range(K1):
                index = ind + k
                HN = H[index]
                # HN = np.convolve(HN, kernel, 'same')
                HN = HN / HN.max()
                if k == 0:
                    H0 = HN
                else:
                    # H0 = np.maximum(H0, HN)
                    H0 += HN
                total_heads += 1
        if i == 0:
            onsets = H0
        else:
            onsets = onsets + H0
        # times+=time()-t0
        # H0 = H0[:-(rm_win-1)] - running_mean(H0, rm_win)
        # H0 = np.array([0 if i < 0 else i for i in H0])
        # H0=H0/H0.max()
        peaks = pick_onsets(H0, threshold=drums[i].get_threshold() + thresholdAdj)
        # remove extrahits used to level peak picking algorithm:
        peaks = peaks[np.where(peaks < filt_spec.shape[0] - 1)]
        drums[i].set_hits(peaks)
        # peaks = madmom.features.onsets.peak_picking(H0, drums[i].get_threshold()+thresholdAdj)
        # if i in [0,1,2]:
        #     picContent.append([H0[:], pick_onsets(H0[:], threshold=drums[i].get_threshold())])
        #     kernel = np.hanning(8)

        # showEnvelope((H0, peaks, drums[i].get_threshold()))

        # showFFT(np.outer(Wpre.T[0, i, :],H0))

        # onsets[peaks] = 1
        # quant_factor > 0:
        #    TEMPO = DEFAULT_TEMPO
        #   qPeaks = timequantize(peaks, avgTempo, TEMPO)
        # qPeaks = quantize(peaks, tempomask, strength=quant_factor, tempo=TEMPO, conform=False)
        # qPeaks=qPeaks*changeFactor
        # else:

    # sanity check
    if False:
        allPeaks.extend(peaks)
        # detect peaks in the full spectrogram, compare to detection results for a sanity check
        sanityspec = get_preprocessed_spectrogram(liveBuffer, sm_win=8, test=True)
        H0 = superflux(spec_x=filt_spec.T, win_size=8)
        HS = sanityspec[:, 0]
        HS = HS / HS[3:].max()
        sanitypeaks = pick_onsets(H0, delta=0.02)
        print(sanitypeaks.shape, len(allPeaks))
        for i in sanitypeaks:
            if np.argwhere(allPeaks == i) is None:
                print('NMFD missed an onset at:', i)

    # duplicate cleaning
    if False:
        duplicateResolution = 0.05
        for i in drums:
            precHits = frame_to_time(i.get_hits())
            i.set_hits(time_to_frame(cleanDoubleStrokes(precHits, resolution=duplicateResolution)))

    if quant_factor > 0:
        onsets = onsets / onsets.max()
        deltaTempo = extract_tempo(onsets, constant_tempo=False)
        newmax=0
        for i in drums:
            hits = i.get_hits()
            if len(hits) is None or len(hits)==0:
                i.set_hits([])
            else:
                i.set_hits(conform_time(hits, deltaTempo))
                if newmax<max(i.get_hits()):
                    newmax=max(i.get_hits())

        #TESTING librosa implementation of Ellis's dp beat detection, Improves quantization!!
        if True:
            beats = np.zeros(newmax+1)
            for i in drums:
                #create an odf from already found onsets weigh toward kick and snare
                beats[i.get_hits()] += 1 - (i.get_name()[0] / (nrOfDrums))
            trimmed_beats=np.trim_zeros(beats)
            tracked_beats=beatSimpleDP(trimmed_beats, alpha=100)
            beat_interval=round((tracked_beats.max()-tracked_beats.min())/tracked_beats.size)
            #print(beat_interval)
            fixed_beats=range(tracked_beats.min(),tracked_beats.max()+100,int(beat_interval))
            beat_diff=np.zeros(tracked_beats.size)
            for i in range(tracked_beats.size):
                beat_diff[i]=fixed_beats[i]-tracked_beats[i]
            for i in drums:
                quant_hits=[]
                for j in i.get_hits():
                    a= (np.abs(tracked_beats -j)).argmin()
                    quant_hits.append(j+int(beat_diff[a]))
                i.set_hits(quant_hits)
        return drums, np.mean(deltaTempo)
    else:
        return drums, 1.0

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




def get_preprocessed_audio(buffer, sr=44100, window_size_in_ms=24, bark_bands=24):
    from scipy.signal import filtfilt, butter, lfilter
    buffer = madmom.audio.signal.normalize(buffer)
    bark_frequencies = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
                        2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000,
                        15500]

    # From scipy cookbooks
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def get_filtered_band(buffer, low, hi, fs):
        b, a = butter_bandpass(low, hi, fs, order=5)
        return filtfilt(b, a, buffer)

    sub_band_audio = np.zeros((buffer.shape[0], bark_bands))
    mu = 10 ** 8
    kernel = np.hanning(3)

    sub_band_audio = madmom.audio.spectrogram.FilteredSpectrogram(buffer, filterbank=proc2, sample_rate=SAMPLE_RATE,
                                                                  frame_size=1024, hop_size=256, fmin=20,
                                                                  fmax=17000)  # , window=np.kaiser(FRAME_SIZE, np.pi ** 2))

    for i in range(20):
        # sub_band_audio[:,i]=butter_bandpass_filter(buffer, bark_frequencies[i],bark_frequencies[i+1],sr)
        sub_band_audio[:, i] = np.log(1 + mu * np.abs(sub_band_audio[:, i])) / np.log(1 + mu)
        # sub_band_audio[:, i] = ((np.sign(sub_band_audio[:,i]) * np.log(1 + mu * np.abs(sub_band_audio[:,i])))) / (1 + np.log(mu))
        # sub_band_audio[:, i] = np.gradient(sub_band_audio[:, i], axis=0)
        # sub_band_audio[:, i] = np.clip(sub_band_audio[:, i], 0, None, out=sub_band_audio[:, i])
        # sub_band_audio[:, i] = np.mean(sub_band_audio[:, i])
        # sub_band_audio[:, i] = np.convolve(sub_band_audio[:, i], kernel, 'same')
        # sub_band_audio[:-1, i] = np.diff(sub_band_audio[:, i],n=1, axis=0)
        # np.clip(sub_band_audio[:, i], 0, None, out=sub_band_audio[:, i])
    bat_odf = np.zeros(sub_band_audio.shape[0])

    for i in range(sub_band_audio.shape[0]):
        bat_odf[i] = np.mean(sub_band_audio[i, :])
    # bat_odf = np.convolve(bat_odf[:-1], kernel, 'same')
    bat_odf[:-1] = np.diff(bat_odf, n=1, axis=0)

    np.clip(bat_odf, 0, None, out=bat_odf)

    bat_odf = bat_odf / max(bat_odf[2:-2])

    # showEnvelope(bat_odf[2:-2])
    return bat_odf[:-2]
    # showEnvelope(sub_band_audio[800000:900000])


def get_preprocessed_spectrogram(buffer=None, A=None, B=None, sm_win=4, test=False, Print=False):
    """
    Preprocess source audio data and return a processed stft

    :param buffer: numpy array, None, source audio
    :param A: numpy array, None, frequency vector of separated data
    :param B: numpy array, None, activations of separated data
    :param sm_win: int, smoothing window size
    :param test: boolean, if true E.Battenberg preprocessing is performed.

    :return: numpy array, preprocessed stft of the source data
    """
    if buffer is not None:
        spec = madmom.audio.spectrogram.FilteredSpectrogram(buffer, filterbank=proc, sample_rate=SAMPLE_RATE,
                                                            frame_size=FRAME_SIZE, hop_size=HOP_SIZE, fmin=20,
                                                            fmax=17000, window=np.kaiser(FRAME_SIZE, np.pi ** 2))

        # if Print:
        #    showEnvelope((buffer[600000:1100000]))
        #    pass

    if A is not None:
        spec = np.outer(A, B).T
    # kernel=np.kaiser(6,5)
    if test:
        mu = 10 ** 8
        for i in range(spec.shape[1]):
            spec[:, i] = np.log(1 + mu * np.abs(spec[:, i])) / np.log(1 + mu)
            # spec[:, i] = ((np.sign(spec[:, i]) * np.log(1 + mu * np.abs(spec[:, i])))) / (1 + np.log(mu))

    kernel = np.hanning(sm_win)
    for i in range(spec.shape[1]):
        spec[:, i] = np.convolve(spec[:, i], kernel, 'same')

    if test:
        spec = np.gradient(spec, axis=0)
        spec = np.clip(spec, 0, None, out=spec)
        # spec = (spec + np.abs(spec)) / 2
        for i in range(spec.shape[0]):
            spec[i, :] = np.mean(spec[i, :])
        # spec[1:]=spec[1:]-spec[:-1]
        # spec=(spec+np.abs(spec))/2
        # spec=spec/spec.max()

    return spec


# superlux from madmom (Boeck et al)
def superflux(spec_x=[], A=None, B=None, win_size=8):
    """
    Calculate the superflux envelope according to Boeck et al.

    :param spec_x: optional, numpy array, A Spectrogram the superflux envelope is calculated from, X
    :param A: optional, numpy array, frequency response of the decomposed spectrogram, W
    :param B: optional, numpy array, activations of a decomposed spectrogram, H
    :param win_size: int, Hann window size, used to smooth the recomposition of

    :return: Superflux ODF of a spectrogram
    :notes: Must check inputs so that A and B have to be input together
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


def frame_to_time(frames, sr=SAMPLE_RATE, hop_length=HOP_SIZE):
    """
    Transforms frame numbers to time values

    :param frames: list of integers to transform
    :param sr: int, Sample rate of the FFT
    :param hop_length: int, Hop length of FFT

    :return: Numpy array of time values
    """

    samples = (np.asanyarray(frames) * (hop_length)).astype(int)
    return np.asanyarray(samples) / float(sr)


def time_to_frame(times, sr=SAMPLE_RATE, hop_length=HOP_SIZE):
    """
    Transforms time values to frame numbers

    :param times: list of timevalues to transform
    :param sr: int, Sample rate of the FFT
    :param hop_length: int, Hop length of FFT

    :return: Numpy array of frame numbers
    """
    samples = (np.asanyarray(times) * float(sr))
    return np.rint(np.asanyarray(samples) / (hop_length))

def time_to_grid(times, sr=SAMPLE_RATE, hop_length=HOP_SIZE):
    """
    Transforms time values to frame numbers

    :param times: list of timevalues to transform
    :param sr: int, Sample rate of the FFT
    :param hop_length: int, Hop length of FFT

    :return: Numpy array of frame numbers
    """
    full=hop_length
    triplet=hop_length*2/3.
    dotted=hop_length*3/2.
    #Pick closest grid and put 'm there....
    samples = (np.asanyarray(times) * float(sr))
    return np.rint(np.asanyarray(samples) / (hop_length))

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
    bpms = 60 * fps / np.array(range(0,max_bpm))

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

    # tic = time()
    # #The tempo extraction method from "Beat Tracking by Dynamic Programming" by Daniel P.W. Ellis
    # #Assumes constant tempo, not useful for this application.
    # #This counts a tempo/frame, very slow! Even if we count tempo per every N frame it is 20 times
    # #slower than our FFT method. Roughly as slow as the STFT method.
    # taus = np.array(list(range(40,180)))
    # ac_tempi=np.zeros(fonsets.shape[1])
    # for j in range(fonsets.shape[1]):
    #     #print(fonsets[:,j].shape)
    #     tempo = []
    #     for tau in taus:
    #         tempo.append(np.exp(-0.5 * ((np.log2(tau) - np.log2(60*fps/DEFAULT_TEMPO)) / 1.4) ** 2)*np.sum(np.abs(fonsets[:,j][tau:] * fonsets[:,j][0:-tau])))
    #     bpm=60*fps/taus
    #     ac_tempi[j]=bpm[np.argmax(tempo)]
    # print('\ntau_global time:{}'.format(time() - tic))
    #showEnvelope(tempi)
    # #STFT based tempogram and autocorrelation tests, SLOW!!!
    # f,t,stft=scipy.signal.stft(fonsets,fs=44100, window='hamming',nperseg=max_bpm, noverlap=None, nfft=max_bpm
    #                   ,detrend=False, return_onesided=False, boundary='zeros', padded='True',axis=0)
    # #print(stft.shape, stft[0])
    # stft=np.sum(stft,axis=2)
    # #showFFT(np.abs(stft))
    # comp_stft = stft * np.conj(stft)
    # #showFFT(np.abs(comp_stft))
    # t,autocorr_stft = scipy.signal.istft(comp_stft)
    # sff_stft=(autocorr_stft.real / autocorr_stft.real.max())
    # sff_stft.resize(max_bpm*onsets.shape[0])
    # sff_stft=np.reshape(sff_stft,[max_bpm,-1])
    # print('sff_stft:', sff_stft.shape)
    # print('\ntau_global time2:{}'.format(time() - tic))



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


def movingAverage(x, window=500):
    return median_filter(x, size=(window))


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
    if print:
        showEnvelope([F, Tdyn])
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


def NMFD(X, iters=128, Wpre=[], include_priors=False,n_heads=1, semi_adaptive=False, beta=4, partially_fixed=False, add_h=0, fully_adaptive=False, random_init=False, hand_break=True):
    """

    :param X:  numpy array, The spectrogram
    :param iters: int, maximum number of iterations
    :param Wpre: numpy array, prior basis matrix
    :param include_priors: boolean, include training data to analysis to avoid false activations
    if no hits of a certain drum is present in the data.
    :param n_heads: int, number of Wpre heads
    :param semi_adaptive: boolean, use semi adaptive updates for W
    :param beta: beta of SA-NMFD
    :param partially_fixed: DO NOT USE, work in progress.
    :param add_h: int, add extra random basis vectors
    :param fully_adaptive: boolean, adapt W every iteration
    :param random_init: boolean, initialize W with random values
    :param hand_break:
    :return: numpy array, numpy array, int, H(Activations of different drums in X), W(basis matrix), normalized errors at iterations

    """
    # epsilon, added to matrix elements to ensure non-negativity in matrices
    eps = 10 ** -18
    #X=X/X.max()
    #remove tails
    #Wpre=Wpre[:, :n_heads, :]
    # Add sound check samples to the audio to prevent normalization error when no hits of certain drums are present
    # The multiplier is arbitrarily chosen, lower than .5 ensures that the quieter hits are not lost
    # as the soundcheck hits are possibly very loud compared to actual playing
    if include_priors:
        for i in range(int(Wpre.shape[1] / 2)):
            pass
            X = np.hstack((X, .28 * Wpre[:, i, :]))
    #remove prior templates
    if random_init:
        split=0
        if partially_fixed:
            split=n_heads
        Wpre[:,split:,:]=np.random.rand(Wpre.shape[0],Wpre.shape[1]-split, Wpre.shape[2])

    # add noise priors for partially_fixed, DO NOT USE, minimal improvement with serious speed drop.
    if partially_fixed or add_h > 0:
        W_z = np.random.rand(Wpre.shape[0], add_h, Wpre.shape[2])
        Wpre = np.concatenate((Wpre, W_z), axis=1)
    # data size
    M, N = X.shape
    W = Wpre
    M, R, T = W.shape
    # Initial H, non negative, non negative. any non negative value works
    H = np.full((R, N), .3)

    # Make sure the input is normalized
    W=W-W.min()
    W = W / W.max()
    X=X-X.min()
    X = X / X.max()
    # all ones matrix
    repMat = np.ones((M, N))

    # Spaceholder for errors
    err = np.zeros(iters)

    #Partially fixed coefficients:
    pf_alpha = R/(n_heads)
    pf_beta = (R-n_heads)/R

    # KLDivergence for cost calculation
    def KLDiv(x, y):
        return (x * np.log(x/y) - x + y).sum()

    # Itakura-Saito Divergence
    def ISDiv(x,y):
        return (x/y - np.log(x/y) - 1).sum()

    # Itakura-Saito Divergence
    # precomputed division, gives a little different results, only for the better
    def ISDiv2(x):
        return (x - np.log(x) - 1).sum()

    # Lambda calculation
    def sumLambda(W, H):
        W_temp=W.copy()
        if(partially_fixed):
            W_temp[:,:n_heads,:]=pf_alpha*W_temp[:,:n_heads,:]
            W_temp[:, n_heads:, :]=pf_beta*W_temp[:,n_heads:,:]
        Lambda = np.zeros((W_temp.shape[0], H.shape[1]))
        shifter = np.zeros((R, N + T + 1))
        shifter[:, T:-1] = H
        for t in range(T):
            Lambda += W_temp[:, :, t] @ shifter[:, T - t: -(t + 1)]
        return Lambda/Lambda.max()

    mu =0.8 # sparsity constraint 0.8 in use mode

    for i in range(iters):

        # Stop processing and spit out result if True
        if REQUEST_RESULT:
            return H, W, err / err.max()

        # Initialize Lambda and Lambda^
        if i==0:
            Lambda = sumLambda(W, H)+eps
            LambHat =X/Lambda + eps

        shifter = np.zeros((M, N + T))
        # Lambhat with T columns of zero padding
        shifter[:, :-T] = LambHat
        Hhat1 = np.zeros((R, N))
        Hhat2 = np.zeros((R, N))

        # the convolution calculation
        for t in range(T):
            Wt = W[:, :, t].T
            # Numerator calc
            Hhat1 += Wt @ shifter[:, t: t + N]
            # Denominator calc
            Hhat2 += Wt @ repMat + eps

        H = H * Hhat1 / (Hhat2 + mu)

        # precompute for error and next round
        Lambda = sumLambda(W, H)+eps
        LambHat = X/Lambda + eps

        # 2*eps to ensure the eps don't cancel out each other in the division.
        #err[i] = KLDiv(X + eps, Lambda + eps)
        #err[i] = ISDiv(X+eps,Lambda+eps)
        err[i]=ISDiv2(LambHat+eps)#+np.linalg.norm(H,ord=np.inf)*mu

        # Stopping criteria check
        if (i >= 1):
            errDiff = (abs(err[i] - err[i - 1]) / (err[0] - err[i] +eps))

            #print(errDiff)
            if errDiff < 0.0005 or err[i] > err[i - 1]:
                if hand_break:
                    break
                pass

        # Adaptive W - no considerable improvement double running time
        if semi_adaptive or partially_fixed or fully_adaptive:
            #adapt everything
            split = 0
            #fully adaptive alpha
            alpha = 0
            if (semi_adaptive):
                alpha = (1 - i / iters) ** beta
            if (partially_fixed):
                split=n_heads
            shifter = np.zeros((N + T, R))
            shifter[:-T, :] = H.T[:, :]

            for t in range(T):
                W[:, split:, t] = W[:, split:, t] * (
                        LambHat @ shifter[t:-(T - t), split:] / (repMat @ shifter[t:-(T - t), split:] + eps + (mu / T)))

            if partially_fixed:
                W = Wpre * (1-alpha) + (alpha) * W
            else:
                W = Wpre * alpha + (1 - alpha) * W

            W = W / W.max()

    return H, W, err / err.max()


#
#


##TO USE NEXT TWO FUNCTIONS UNCOMMENT matplotlib IMPORTS; CONFLICTS WITH KIVY
def showEnvelope(env, legend=None, labels=None):
    """
    Plots an envelope
    i.e. an onset envelope or some other 2d data
    :param env: the data to plot
    :return: None
    """
    if 0 > 1:
        f, axarr = plt.subplots(len(env), 1, sharex=True, sharey=True)
        #for i in range(len(env)):
        #    axarr[i].plot(env[i][0], label='NMF')
        #    axarr[i].vlines(env[i][1], 0, 1, color='r', alpha=0.9, linestyle='--', label='Onsets')
        #    axarr[i].get_xaxis().set_visible(False)
        #    axarr[i].get_yaxis().set_ticks([])
        #f.subplots_adjust(hspace=0.1)
        axarr[0].set(ylabel='precision')
        axarr[1].set(ylabel='recall')
        axarr[2].set(ylabel='f-score')
        axarr[len(env) - 1].get_xaxis().set_visible(True)
        plt.savefig("NMFD_test_iterations.png")
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 6))

        # plt.ylim(ymax=1)
        plt.plot(env)
        # plt.plot(env[0], label='Onset envelope')
        #
        # #plt.hlines(env[2],0,8000, color='r', linestyle=':')
        # plt.hlines(env[2], 0,500, color='r', alpha=0.8, label='threshold', linestyles='-')
        # plt.hlines(env[3], 0, 500, color='k', alpha=0.8, label='highest optimal threshold', linestyles='--')
        # plt.hlines(env[1], 0, 500, color='g', alpha=0.8, label='lowest optimal threshold', linestyles=':')
        #
        #plt.xticks(np.geomspace(1, 50, 5).astype(int), np.geomspace(1, 50, 5).astype(int),1)
        plt.gca().legend(('precision','recall','f-score'), loc='right')
        plt.xlabel(labels[0], fontsize=12)
        plt.ylabel(labels[1], fontsize=12)
        plt.show()
        #ax = plt.subplot(111)
        #if legend != None:
        #    box = ax.get_position()
        #    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #    plt.gca().legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
        #    plt.gca().legend(legend, loc='right')
        #if labels != None:
        #    # my_xticks=[str(x)[:4] for x in np.linspace(0.2,0.4,int(env.shape[0]/2))]
        #    # my_xticks = np.arange(0, env.shape[0], .01)
        #    # plt.xticks(np.arange(1, env.shape[0], 1), range(1,20))
#
        #    plt.xlabel(labels[0], fontsize=12)
        #    plt.ylabel(labels[1], fontsize=12)
#
        # plt.savefig("nadam_lr.png")
        # plt.tight_layout()


def showFFT(env, ticks=None, stft=False):
    """
    Plots a spectrogram

    :param env: the data to plot
    :return: None
    """
    if 0 > 1:

        f, axarr = plt.subplots(1, len(env), sharex=True, sharey=True)

        for i in range(len(env)):
            axarr[i].imshow(env[i], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))

            # axarr[i].get_xaxis().set_visible(False)
        # axarr[1].imshow(env[1], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
        # axarr[1].set(xlabel='Snare frames')
        # axarr[2].imshow(env[2], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
        # axarr[2].set(xlabel='Closed Hi-Hat frames')
        f.subplots_adjust(wspace=0.03)
        axarr[0].set(ylabel='STFT bin')
        f.text(0.5, 0.02, 'k', ha='center', va='center')

        plt.savefig("Templates3.png")

    if ticks != None:
        top = len(ticks)
        plt.imshow(np.flipud(env), aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
        my_yticks = ticks
        plt.xlabel('frame', fontsize=12)
        plt.ylabel('tempo', fontsize=12)
        plt.yticks(np.arange(0, top, 10), np.rint(np.fliplr([ticks[0:top:10], ]))[0])

    else:
        plt.figure(figsize=(10, 3))

        plt.imshow(env, aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
        plt.show()
    # plt.xlabel('frame', fontsize=12)
    # plt.ylabel('stft bin', fontsize=12)
    # # f, axarr = plt.subplots(3, sharex=True, sharey=True)
    # plt.subplot(131, sharex=True, sharey=True)
    # plt.imshow(env[0], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
    # plt.subplot(132)
    # plt.imshow(env[1], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
    # plt.subplot(133)


def acceptHit(value, hits):
    """
    Helper method to clear mistakes of the annotation such as ride crash and an open hi hat.
    :param value: int the hit were encoding
    :param hits: binary string, The hits already found in that location
    :return: boolean, if it is ok to annotate this drum here.
    """
    # Test to discard mote than 3 drums on the same beat
    sum = 0
    for i in range(value):
        if np.bitwise_and(i, hits):
            sum += 1
        if sum > 3:
            return False
    # discard overlapping cymbals for ease of annotation and rare use cases of overlapping cymbals.
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
    """
    Enceode consecutive pauses in notation to negative integers
    :param frames: numpy array, the notation of all frames
    :return: numpy array, notation with pause frames truncated to neg integers
    """
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
    # print(len(a[-1]))
    if (len(a) == 0):
        return []
    maxFrame = int(a[-1][0] - a[0][0] + 1)
    # print(maxFrame)
    # spaceholder for return array
    frames = np.zeros(maxFrame, dtype=int)
    for i in range(len(a)):
        # define true index by substracting the leading empty frames
        index = int(a[i][0] - a[0][0])
        if index >= frames.shape[0]:
            break
        # The actual hit information
        value = int(a[i][1])
        # Encode the hit into a character array, place 1 on the index of the drum #
        if acceptHit(value, frames[index]):
            try:
                frames[index] = np.bitwise_or(frames[index], 2 ** value)
            except:
                print(frames[index], value)

    # return array of merged hits starting from the first occurring hit event
    if ENCODE_PAUSE:
        frames = truncZeros(frames)
    # print('frames',len(frames))
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
    # print(deltaTempo)
    frameMul = 1 / deltaTempo
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


def semi_adaptive_NMFB(X, Wpre, iters=100,n_heads=1, semi_adaptive=False, beta=4,partially_fixed=False,add_h=0, div=1, hand_break=True):
    epsilon = 10 ** -18
    X = X + epsilon
    #add noise priors
    Wpre=Wpre[:,:n_heads,:]
    if partially_fixed or add_h > 0:
        W_z = np.random.rand(Wpre.shape[0], add_h, Wpre.shape[2])
        Wpre = np.concatenate((Wpre, W_z), axis=1)
    # Use only the first frame of NMFD prior
    W =Wpre[:,:,0]
    M,R=W.shape[0], W.shape[1]
    # Initial H, non negative, non zero. any non zero value works
    H = np.full((W.shape[1], X.shape[1]), .5)
    # normalize input
    W = W / W.max()
    X = X / X.max()
    # error space
    err = np.zeros(iters)

    def KLDiv(x, y):
        return sum(sum(x * np.log(x / y+epsilon) + (y - x)))
    def ISDiv(x, y):
        return ((x / y + epsilon) - np.log(x / y + epsilon) - 1).sum()

    mu = 0.8
    for i in range(iters):
        if partially_fixed:
            pf_alpha = R / (n_heads)
            pf_beta = (R - n_heads) / R
            Wt_alpha = W[:, :n_heads].T
            Wt_beta = W[:, n_heads:].T
            WH_alpha = pf_alpha * W[:, :n_heads] @ H[:n_heads, :]
            WH_beta = pf_beta * W[:, n_heads:] @ H[n_heads:, :]
        else:
            Wt = W.T
            WH = epsilon + np.dot(W, H)
        if partially_fixed:
            H_alpha= (Wt_alpha @ (X * (WH_alpha+WH_beta) ** (div- 2)))/(epsilon + (Wt_alpha @ (WH_alpha+WH_beta) ** (div- 1)))
            H_beta= (Wt_beta @ (X * (WH_alpha+WH_beta) ** (div- 2)))/(epsilon + (Wt_beta @ (WH_alpha+WH_beta) ** (div- 1)))
            H[:n_heads,:]   =  H[:n_heads,:]   * H_alpha+ mu
            H[n_heads:,:]  =  H[n_heads:,:]  * H_beta+ mu

        else:
            H= H*((Wt @ (X * WH ** (div- 2)))/(epsilon + (Wt @ WH ** (div- 1)))+ mu)

        if partially_fixed:
            WH_alpha=pf_alpha*W[:,:n_heads]@H[:n_heads,:]
            WH_beta=pf_beta*W[:,n_heads:]@H[n_heads:,:]
        else:
            WH = epsilon + np.dot(W, H)
        if partially_fixed:
            if div==1:
                err[i] = KLDiv(X + epsilon, (WH_alpha+WH_beta) + epsilon)
            else:
                err[i] = ISDiv(X + epsilon, (WH_alpha+WH_beta) + epsilon)
        else :
            if div==1:
                err[i] = KLDiv(X + epsilon, WH + epsilon)  # + (sparsity * normH)
            else:
                err[i] = ISDiv(X + epsilon, WH + epsilon)
        if (i >= 1):
            errDiff = (abs(err[i] - err[i - 1]) / (err[1] - err[i] + epsilon))
            if errDiff < 0.003 or err[i] > err[i - 1]:
                pass
                if hand_break:
                    break

        if semi_adaptive or partially_fixed:
            alpha=0
            if (semi_adaptive):
                alpha = (1 - i / iters) ** beta
            if partially_fixed:
                Ht=H[n_heads:,:].T
                Wu = (X * WH_beta ** (div - 2)@Ht)
                Wd= (WH_beta ** (div - 1))@Ht+epsilon
                W[:, n_heads:]=(Wt_beta.T*(Wu/(Wd))+mu)
            else:
                Ht=H.T
                Wu = (X * WH ** (div - 2)@Ht)
                Wd=(WH ** (div - 1))@Ht+epsilon
                W=W*(Wu/(Wd))+mu
            if semi_adaptive:
                W = Wpre[:,:,0] * alpha + (1 - alpha) * W

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


def PSA(X, fpr):
    eps = 10 ** -18
    X = X.T
    # fpr = fpr[:, :, 0]
    print(fpr.shape)
    fpr = fpr / fpr.max()
    fpp = np.linalg.pinv(fpr)
    print(fpp.shape)
    tHat = (fpp @ X).T
    # for i in range(9):
    #    showEnvelope(tHat.T[i])
    print(tHat.shape)
    ica = FastICA(n_components=9)
    t = (ica.fit_transform(tHat))
    print(t.shape)
    tp = np.linalg.pinv(t)
    f = X @ tp.T
    print(f.shape)
    f = f / f.max()

    def KLDiv(x, y):
        return (x * (np.log(y / x + eps) - x + y)).sum()

    def ISDiv(x, y):
        return (y / x - np.log(y / x) - 1).sum()

    retf = np.zeros_like(f)
    print(fpr.shape, f.shape)
    for i in range(f.shape[1]):
        x = fpr.T
        y = f.T

        y = abs(y)
        mini = ISDiv(x[i], y[0])
        # print(x[i], y[0])
        index = 0
        for j in range(y.shape[0] - 1):
            if mini >= ISDiv(x[i], y[j]):
                mini = ISDiv(x[i], y[j])
                index = j
        print(index, mini)
        retf.T[i] = y[index]
    tTilde = (np.linalg.pinv(f) @ X).T
    tTilde = np.clip(np.abs(tTilde.T), 0, None)
    showEnvelope(tTilde.T)
    for i in range(9):
        showEnvelope(tTilde[i] / tTilde[i].max())
    return t, f
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


def plp(autocorr, bpm):
    """
    unfinished business
    :param autocorr:
    :param bpm:
    :return:
    """
    tau = autocorr[60:240]
    print(np.argmax(autocorr[:, 10]))
    tempi = np.argmax(np.abs(tau), axis=0)
    # print([tau[tempi[i],i] for i in range(tempi.shape[0])], tau.shape)

    # showFFT(np.abs(autocorr))
    omega_t = np.angle([tau[tempi[i], i] for i in range(tempi.shape[0])])
    # Tperiod = parameter.featureRate* 60. / BPM(local_max(frame))
    # cosine = window * cos((0:1 / Tperiod:len-1 / Tperiod) * 2 * pi + omega_t)
    print((omega_t[:10]))
    # showFFT(np.abs(omega_t))


def hann_poisson_window(N=8, alpha=0.2):
    """
    Hann-Poisson window

    :param N: int, window size
    :param alpha: float, alpha value

    :return: numpy array, Hann-Poisson window
    """
    window = np.zeros(N)

    for n in range(1, N):
        window[n] = .5 * (1 - np.cos((2 * np.pi * n) / n - 1)) * np.exp(-alpha * (np.abs(N - 1 - 2 * n) / (N - 1)))

    return window

def findDefBinsDBSCAN(frames, filteredSpec, ConvFrames, eps=0.5):
    """
    Calculate the prior vectors for W to use in NMF
    :param frames: Numpy array of hit locations (frame numbers)
    :param filteredSpec: Spectrogram, the spectrogram where the vectors are extracted from
    :param ConvFrames: int, number of frames the priors contain
    :return: tuple of Numpy arrays, prior vectors Wpre,heads for actual hits and tails for decay part of the sound
    """
    from sklearn.cluster import DBSCAN
    global total_priors
    eps=eps
    gaps = np.zeros((frames.shape[0], max_n_frames))
    # gaps = np.zeros((frames.shape[0], ConvFrames))
    for i in range(frames.shape[0]):
        for j in range(ConvFrames):
            gaps[i, j] = frames[i] + j

    a = np.reshape(filteredSpec[gaps.astype(int)], (nrOfPeaks, -1))
    dbs = DBSCAN(eps=eps, min_samples=4).fit(a)
    K1,unique_labels=np.unique(dbs.labels_,return_inverse=True)
    indices=np.unique(unique_labels)
    heads = np.zeros((proc.shape[1], max_n_frames, K1.shape[0]))
    for i in indices:
        heads[:, :, i] = np.reshape(np.mean(a[unique_labels==i, :],axis=0), (proc.shape[1], max_n_frames), order='F')
    eps=eps
    tailgaps = np.zeros((frames.shape[0], max_n_frames))
    for i in range(frames.shape[0]):
        for j in range(tailgaps.shape[1]):
            tailgaps[i, j] = frames[i] + j + ConvFrames
    a2 = np.reshape(filteredSpec[tailgaps.astype(int)], (nrOfPeaks, -1))
    dbs = DBSCAN(eps=eps, min_samples=4).fit(a2)
    K2, unique_labels = np.unique(dbs.labels_, return_inverse=True)
    indices = np.unique(unique_labels)
    tails = np.zeros((proc.shape[1], max_n_frames, K2.shape[0]))
    for i in indices:
        tails[:, :, i] = np.reshape(np.mean(a2[unique_labels == i, :], axis=0), (proc.shape[1], max_n_frames),
                                    order='F')
    total_priors += K1.shape[0] + K2.shape[0]
    print(K1.shape[0]+K2.shape[0])
    return (heads, tails, K1, K2)

def findDefBinsOPTICS(frames=None, filteredSpec=None, ConvFrames=None, matrices=None):
    """
    Calculate the prior vectors for W to use in NMF
    :param frames: Numpy array of hit locations (frame numbers)
    :param filteredSpec: Spectrogram, the spectrogram where the vectors are extracted from
    :param ConvFrames: int, number of frames the priors contain
    :return: tuple of Numpy arrays, prior vectors Wpre,heads for actual hits and tails for decay part of the sound
    """
    #from sklearn.cluster import DBSCAN
    from python import OPTICS
    global total_priors
    if matrices is None:
        gaps = np.zeros((frames.shape[0], max_n_frames))
        # gaps = np.zeros((frames.shape[0], ConvFrames))
        for i in range(frames.shape[0]):
            for j in range(ConvFrames):
                gaps[i, j] = frames[i] + j
        a = np.reshape(filteredSpec[gaps.astype(int)], (nrOfPeaks, -1))
        print(a.shape)
    else:
        a=np.array(matrices[0])
        #a=np.reshape(a,(a.shape[0], -1))
        #print(a.shape)

    opts=OPTICS.OPTICS(min_samples=5, max_eps=np.inf, metric='l2',maxima_ratio=0.5, rejection_ratio=2,leaf_size=32).fit(a)
    K1,unique_labels=np.unique(opts.labels_,return_inverse=True)
    indices=np.unique(unique_labels)
    heads = np.zeros((proc.shape[1], max_n_frames, K1.shape[0]))
    for i in indices:
        heads[:, :, i] = np.reshape(np.mean(a[unique_labels==i, :],axis=0), (proc.shape[1], max_n_frames), order='F')
    if matrices is None:
        tailgaps = np.zeros((frames.shape[0], max_n_frames))
        for i in range(frames.shape[0]):
            for j in range(tailgaps.shape[1]):
                tailgaps[i, j] = frames[i] + j + ConvFrames
        a2 = np.reshape(filteredSpec[tailgaps.astype(int)], (nrOfPeaks, -1))
    else:
        a2 = np.array(matrices[1])
        #a2 = np.reshape(a2, (a2.shape[0], -1))
        #print(a2.shape)
    opts = OPTICS.OPTICS(min_samples=5, max_eps=np.inf, metric='l2',maxima_ratio=0.5, rejection_ratio=2,leaf_size=32).fit(a2)
    K2, unique_labels = np.unique(opts.labels_, return_inverse=True)
    indices = np.unique(unique_labels)
    tails = np.zeros((proc.shape[1], max_n_frames, K2.shape[0]))
    for i in indices:
        tails[:, :, i] = np.reshape(np.mean(a2[unique_labels == i, :], axis=0), (proc.shape[1], max_n_frames),
                                    order='F')
    total_priors += K1.shape[0] + K2.shape[0]
    print(K1.shape[0]+K2.shape[0])
    return (heads, tails, K1, K2)

def findDefBins(frames=None, filteredSpec=None, ConvFrames=None, matrices=None):
    """
    Calculate the prior vectors for W to use in NMF by averaging the sample locations
    :param frames: Numpy array of hit locations (frame numbers)
    :param filteredSpec: Spectrogram, the spectrogram where the vectors are extracted from
    :return: tuple of Numpy arrays, prior vectors Wpre,heads for actual hits and tails for decay part of the sound
     """
    global total_priors
    if matrices is None:
        gaps = np.zeros((frames.shape[0], ConvFrames))
        for i in range(frames.shape[0]):
            for j in range(gaps.shape[1]):
                gaps[i, j] = frames[i] + j
        a = np.reshape(filteredSpec[gaps.astype(int)], (nrOfPeaks, -1))
    else:
        a = np.array(matrices[0])
    heads= np.reshape(np.mean(a, axis=0), (proc.shape[1],max_n_frames, 1), order='F')
    if matrices is None:
        tailgaps = np.zeros((frames.shape[0], ConvFrames))
        for i in range(frames.shape[0]):
            for j in range(gaps.shape[1]):
                tailgaps[i, j] = frames[i] + j + ConvFrames
        a2 = np.reshape(filteredSpec[tailgaps.astype(int)], (nrOfPeaks, -1))
    else:
        a2=np.array(matrices[1])
    tails = np.reshape(np.mean(a2, axis=0), (proc.shape[1], max_n_frames, 1), order='F')
    total_priors += 2
    return (heads, tails, 1,1)
