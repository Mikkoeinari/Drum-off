import madmom
import numpy as np
from scipy.ndimage.filters import maximum_filter, median_filter
import scipy
import librosa
FRAME_SIZE=2**11
HOP_SIZE=2**9
SAMPLE_RATE=44100
FREQUENCY_PRE=np.ones((24))#[0,16384]#0-2^14
MIDINOTE=36 #kickdrum in most systems
THRESHOLD=0.0
PROBABILITY_THRESHOLD=0.0
DRUMKIT_PATH='../DXSamplet/'
midinotes=[36,38,42,46,50,43,51,49,44] #BD, SN, CHH, OHH, TT, FT, RD, CR, SHH
nrOfDrums=9
nrOfPeaks=32
ConvFrames=10
K=1
MS_IN_MIN=60000
SXTH_DIV=8
proc=madmom.audio.filters.BarkFilterbank(madmom.audio.stft.fft_frequencies(num_fft_bins=int(FRAME_SIZE/2), sample_rate=SAMPLE_RATE) ,num_bands='double')


# Tämä on vähän turha luokka, kaiken voi tallettaa drum objektiin
class detector(object):
    def __init__(self, drum, hitlist=None
                 , **kwargs):
        # set attributes
        self.drum = drum
        if hitlist:
            self.hitlist = hitlist
        else:
            self.hitlist = None

    def set_hits(self, hitlist):
        self.hitlist = hitlist

    def get_hits(self):
        return self.hitlist

    def get_name(self):
        return self.drum.get_name()

    def get_midinote(self):
        return self.drum.get_midinote()

    def get_threshold(self):
        return self.drum.get_threshold()

    def get_frequency_pre(self):
        return self.drum.get_frequency_pre()

    def get_probability_threshold(self):
        return self.drum.get_probability_threshold()




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
                 , **kwargs):

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
        if probability_threshold:
            self.probability_threshold = float(probability_threshold)

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_highEmph(self, highEmph):
        self.highEmph = highEmph

    def get_highEmph(self):
        return self.highEmph

    def set_peaks(self, name):
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

from sklearn.cluster import KMeans

def findDefBins(frames, filteredSpec):
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

def getStompTemplate(numHits=2, recordingLength=1, highEmph=0):
    """
    Kutsutaan pari kertaa, eka vaikka kaks ja sit neljä iskua talteen
    Niiden pohjalta lasketaan olennaisin taajuus ja threshold
    """
    stompResolution = 1
    buffer = np.zeros(shape=(227792 * recordingLength))
    j = 0
    strm = madmom.audio.signal.Stream(sample_rate=44100, num_channels=1, frame_size=2048, hop_size=HOP_SIZE)
    for i in strm:
        # print(i.shape)
        buffer[j:j + HOP_SIZE] = i[:HOP_SIZE]
        j += HOP_SIZE
        if j >= 221792 * recordingLength:
            buffer[j:j + 6000] = np.zeros(6000)
            # buffer=madmom.audio.signal.normalize(buffer)
            peaks, bins, threshold = getPeaksFromBuffer(buffer, stompResolution, numHits, highEmph=highEmph)

            strm.close()
            return peaks, bins, threshold, buffer

def getPeaksFromBuffer(buffer, resolution, numHits, highEmph=0):
    filt_spec = get_preprocessed_spectrogram(buffer)
    threshold = 5
    searchSpeed = 0.1
    # peaks=cleanDoubleStrokes(madmom.features.onsets.peak_picking(superflux_3,threshold),resolution)
    H0 = superflux(spec_x=filt_spec.T)
    peaks = cleanDoubleStrokes(pick_onsets(H0, initialThreshold=0.15, delta=threshold), resolution)
    while (peaks.shape != (numHits,)):
        # Make sure we don't go over numHits
        # There is a chance of an infinite loop here!!! Make sure that don't happen
        if (peaks.shape[0] > numHits):
            threshold += searchSpeed
            searchSpeed = searchSpeed / 2
        threshold -= searchSpeed
        # peaks=cleanDoubleStrokes(madmom.features.onsets.peak_picking(superflux_3,threshold),resolution)
        peaks = cleanDoubleStrokes(pick_onsets(H0, initialThreshold=0.15, delta=threshold), resolution)

    definingBins = findDefBins(peaks, filt_spec)
    return peaks, definingBins, threshold


def processLiveAudio(liveBuffer=None, peakList=None, classifier='LGB', basis=None, quant_factor=0.0):

    filt_spec = get_preprocessed_spectrogram(liveBuffer)
    iters = 1.
    for i in range(int(iters)):
        # Xpost, W, H = semi_adaptive_NMFB(filt_spec.T,basis,n_iterations=2000,
        #                             alternate=False, adaptive=False, leeseung='bat', div=0, sp=0)
        Xpost, W, H = NMFD(filt_spec.T, n_iter=500, init_W=basis)
        if i == 0:
            XpostTot, WTot, HTot = Xpost, W, H
        else:
            XpostTot += Xpost
            WTot += W
            HTot += H
    W = (WTot) / iters
    H = (HTot) / iters
    # kernel=np.kaiser(20,0)
    # from scipy.signal import convolve
    # Hf=convolve(np.sum(H, axis=0), kernel, 'same')
    # plt.figure()
    # plt.plot(Hf)
    if quant_factor > 0:
        bdTempo = (librosa.beat.tempo(onset_envelope=np.sum(H, axis=0), sr=SAMPLE_RATE, hop_length=HOP_SIZE, ac_size=8,
                                      aggregate=None))
        bdAvg = movingAverage(bdTempo, window=1200)
        # print(bdAvg.shape)
        # snTempo=(librosa.beat.tempo( onset_envelope=H[1], sr=SAMPLE_RATE,hop_length=HOP_SIZE))
        # plt.figure()
        ##plt.plot(movingAverage(snTempo, 0,bdTempo.shape))
        # plt.plot(bdTempo, color='r')
        # plt.plot(bdAvg, color='g')
        # print(np.mean(bdTempo))
        # print(np.mean(bdAvg))
        # win=2000
        # plt.plot(librosa.beat.tempo(onset_envelope=H[0], sr=SAMPLE_RATE,hop_length=HOP_SIZE
        #                             , aggregate=movingAverage), color='b')
        tempomask = tempoMask(bdAvg)

    for i in range(len(peakList)):

        H0 = superflux(A=W.T[:, i, :].sum(), B=H[i])
        # HFCX0=HFC_filter(Xpost[i.get_name()][0])
        # H0=superflux(W[i.get_name()].T[0],HFCX0)
        # print (peakList[i].get_threshold())
        peaks = pick_onsets(H0, initialThreshold=0.15, delta=2.5)
        # peaks=madmom.features.onsets.peak_picking(H0 ,i.get_threshold(),
        #                                      smooth=None, pre_avg=0, post_avg=0,pre_max=1, post_max=1)
        if quant_factor > 0:
            TEMPO = 120

            qPeaks = quantize(peaks, tempomask, strength=quant_factor, tempo=TEMPO, conform=False)
        else:
            qPeaks = peaks
        # cPeaks=conformToTempo(qPeaks, tempo, tempomask)

        peakList[i].set_hits(qPeaks)

    def mergeKdrums(peakList):
        newPeakList = []
        for i in range(nrOfDrums):
            ind = i * K

            for k in range(K):

                if k == 0:
                    newPeakList.append(peakList[ind + k])

                else:

                    newPeakList[i].set_hits(np.hstack((newPeakList[i].get_hits(), peakList[ind + k].get_hits())))
            a = newPeakList[i].get_hits()
            newPeakList[i].set_hits(np.unique(a[a != np.array(None)]))

        return newPeakList

    finalList = mergeKdrums(peakList)
    duplicateResolution = 0.05

    for i in finalList:
        precHits = frame_to_time(i.get_hits())
        i.set_hits(time_to_frame(cleanDoubleStrokes(precHits, resolution=duplicateResolution)))

    # prune onsets with CNN classifier
    if (classifier == 'CNN'):

        for i in finalList:
            prob_thresh = i.get_probability_threshold()
            i.set_hits(pruneCNN(i.get_hits(), i.get_name(), cnn, prob_thresh, liveBuffer))

    return finalList
# From https://stackoverflow.com/questions/24354279/python-spectral-centroid-for-a-wav-file
def spectral_centroid(x, samplerate=44100):
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
                                                        frame_size=FRAME_SIZE, hop_size=HOP_SIZE, fmin=20)
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
                    rate=44100,
                    output=True)
    # play stream (3)
    f = 0
    print(len(data))
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
#From https://stackoverflow.com/questions/1566936/
class prettyfloat(float):
    def __repr__(self):
        return "%0.3f" % self

#superlux from madmom (Boeck et al)
def superflux(spec_x=[], A=None,B=None):
    if A!=None:
        kernel=np.hamming(8)
        from scipy.signal import convolve
        B=convolve(B, kernel, 'same')
        B=B/max(B)
        spec_x=np.outer(A,B)
    diff = np.zeros_like(spec_x.T)
    size = (2,16)
    max_spec = maximum_filter(spec_x.T, size=size)
    diff[1:] = (spec_x.T[1:] - max_spec[: -1])
    pos_diff = np.maximum(0, diff)
    sf = np.sum(pos_diff, axis=1)
    sf=sf/max(sf)
    return sf


def frame_to_time(frames, sr=44100, hop_length=HOP_SIZE, hops_per_frame=1):
    samples = (np.asanyarray(frames) * (hop_length / hops_per_frame)).astype(int)
    return np.asanyarray(samples) / float(sr)


def time_to_frame(times, sr=44100, hop_length=HOP_SIZE, hops_per_frame=1):
    samples = (np.asanyarray(times) * float(sr))
    return (np.asanyarray(samples) / (hop_length / hops_per_frame)).astype(int)


def f_score(hits, hitNMiss, actual):
    try:
        precision = (float(hits) / hitNMiss)
        recall = (float(hits) / actual)
        fscore = (2 * ((precision * recall) / (precision + recall)))
        return (precision, recall, fscore)
    except Exception as e:
        print (e)
        return (0.0, 0.0, 0.0)


def k_in_n(k, n, window=1):
    hits = 0
    for i in n:
        for j in k:
            if (j - window <= i <= j + window):
                hits += 1
                break
            if (j + window > i):
                break
    return float(hits)


MS_IN_MIN = 60000
SXTH_DIV = 8


# Create a mask of 16th notes for the duration of the drumtake based on the tempomap
def tempoMask(tempos):
    # Move all tempos to half-notes to counter erratic behaviour when tempo extraction doubles tempo value.
    for i in range(tempos.size):
        if tempos[i] > 100:
            while tempos[i] > 100:
                tempos[i] = tempos[i] / 2
    # define the length of a sixteenthnote in ms in relation to tempo at time t
    sixtLength = MS_IN_MIN / tempos / SXTH_DIV
    # define the length of a frame in ms
    frameLength = 1000 * HOP_SIZE / SAMPLE_RATE
    # extract indices in X that correspond to a 16th note in tempo at time t
    # by calculating modulo and taking the inverse boolean array mod(a,b)==0 ==True
    indices = np.array([int((s % (sixtLength[s] / frameLength))) for s in range(0, tempos.shape[0])])
    invertedIndices = np.invert(indices.astype('bool'))
    return invertedIndices


# Quantize hits accordng to 16th Note mask
def quantize(X, mask, strength=1, tempo=120, conform=False):
    # Create a mask of constant tempo
    if conform:
        # drop less than 100 to half
        if tempo < 100:
            tempo = tempo / 2
        conformMask = tempoMask(np.full(mask.size * 2, tempo))
        trueInd = np.where(mask == True)[0]
        dim = min(np.where(conformMask == True)[0].size, trueInd.size)
        cMask = np.where(conformMask == True)[0][:dim] - trueInd[:dim]
        shiftMask = mask.astype(int)
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

    return retX[retX != np.array(None)]


def get_cmask(tempo, tempomask):
    return cMask


#aloita ikkuna x[:-window]
def movingAverage(x,window=500):
        return median_filter(x, size=(window))
from scipy.signal import argrelmax

def pick_onsets(F, initialThreshold=0, delta=0):
    # dynamic threshold from Bello et. al. sigma_hat[T]=initialThreshold+delta(median(window_at_T))
    localMaximaInd=argrelmax(F, order=1) #max indices
    localMaxima=F[localMaximaInd[0]]
    #localMaximaF=Hf[localMaximaInd[0]]
    #Tdyn=initialThreshold+delta*(np.mean(localMaxima))
    #Tdyn=delta*(np.mean(localMaxima))
    #T1, T2, Tdyn=np.zeros(localMaxima.shape[0]),np.zeros(localMaxima.shape[0]),np.zeros(localMaxima.shape[0])
    #for i in range(localMaxima.shape[0]):
    #    n=localMaxima[i:i+50]
    #
    #    T1[i] = 0.15+1.5*(np.percentile(n,75)-np.percentile(n, 25)) + np.percentile(n, 50)
    #    T2[i] =0.1*np.percentile(n, 100)
    #    p=1
    #    Tdyn[i]=((T1[i]**p+T2[i]**p)/2.)**(1./p)
    Tdyn=delta*(np.mean(localMaxima))
    #Tf=np.array(localMaximaF>=1)

    onsets=np.array(localMaxima>=Tdyn)
    #onsets=np.logical_and(onsets,Tf)
    return localMaximaInd[0][onsets]

from scipy import signal
def HFC_filter(X):
    #b, a = signal.butter(4,0.5)
    Xi=np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        Xi+=i*100*X[i]
    plt.figure()
    plt.plot(Xi)
    return (Xi/Xi.max())


import sys


def NMFD(V, R=3, T=10, n_iter=50, init_W=[], init_H=[]):
    """
     NMFD(V, R=3 ,T=10, n_iter=50, init_W=None, init_H=None)
        NMFD as proposed by Smaragdis (Non-negative Matrix Factor
        Deconvolution; Extraction of Multiple Sound Sources from Monophonic
        Inputs). KL divergence minimization. The proposed algorithm was
        corrected.
    Input :
       - V : magnitude spectrogram to factorize (is a FxN numpy array)
       - R : number of templates (unused if init_W or init_H is set)
       - T : template size (in number of frames in the spectrogram) (unused if init_W is set)
       - n_iter : number of iterations
       - init_W : initial value for W.
       - init_H : initial value for H.
    Output :
       - W : time/frequency template (FxRxT array, each template is TxF)
       - H : activities for each template (RxN array)
     Copyright (C) 2015 Romain Hennequin
    """

    """
     V : spectrogram FxN
     H : activation RxN
     Wt : spectral template FxR t = 0 to T-1
     W : FxRxT
    """
    eps = 10 ** -18

    # data size
    F = V.shape[0];
    N = V.shape[1];

    # initialization
    if len(init_H):
        H = init_H
        R = H.shape[0]
    # if W inited R from W
    elif len(init_W):
        H = np.random.rand(init_W.shape[1], N);
    else:
        H = np.random.rand(R, N);

    if len(init_W):
        W = init_W
        R = W.shape[1]
        T = W.shape[2]
    else:
        W = np.random.rand(F, R, T);

    One = np.ones((F, N));
    Lambda = np.zeros((F, N));

    cost = np.zeros(n_iter)

    def KLDiv(x, y):
        return sum(sum(x * np.log(x / y + eps) + (y - x)))

    for it in range(n_iter):
        sys.stdout.write('Computing NMFD. iteration : %d/%d' % (it + 1, n_iter));
        sys.stdout.write('\r')
        sys.stdout.flush()

        # computation of Lambda
        Lambda[:] = 0;
        for f in range(F):
            for z in range(R):
                cv = np.convolve(W[f, z, :], H[z, :]);
                Lambda[f, :] = Lambda[f, :] + cv[0:Lambda.shape[1]];

        # Halt = H.copy();

        # Htu = np.zeros((T,R,N));
        # Htd = np.zeros((T,R,N));

        # update of H for each value of t (which will be averaged)
        VonLambda = V / (Lambda + eps);

        cost[it] = (V * np.log(V / Lambda + eps) - V + Lambda).sum()

        Hu = np.zeros((R, N));
        Hd = np.zeros((R, N));
        for z in range(R):
            for f in range(F):
                cv = np.convolve(VonLambda[f, :], np.flipud(W[f, z, :]));
                Hu[z, :] = Hu[z, :] + cv[T - 1:T + N - 1];
                cv = np.convolve(One[f, :], np.flipud(W[f, z, :]));
                Hd[z, :] = Hd[z, :] + cv[T - 1:T + N - 1];

        # average along t
        H = H * Hu / Hd;

        # cost[it]=KLDiv(V,Lambda+eps)
        if (it >= 2):
            if (abs(cost[it] - cost[it - 1]) / (cost[1] - cost[it]) + eps) < 0.0001:
                # print('Iterations: {}'.format(i))
                # print('Error: {}'.format(err[i]))
                break
    return cost, W, H