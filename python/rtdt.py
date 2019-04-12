import utils
import game
import time
import pyaudio
import nmfd
import onset_detection
import numpy as np
from constants import *
import matplotlib.pyplot as plt
import matplotlib.cm as cmaps
from scipy.signal import istft
def vocoder(data, tone=None):
    def synthesize(fft_data, tone=None):
        istft_data=istft(fft_data, fs=44100, window='hann', nperseg=1024, noverlap=None, nfft=1024, input_onesided=True, boundary=True, time_axis=-2, freq_axis=-1)
        return istft_data
    filt_spec = utils.stft(data, streaming=False, filterbank=c_major_filterbank())
    print(filt_spec.max())
    print(filt_spec.shape)
    idata=synthesize(filt_spec, tone=None)
    print(idata[1].max())
    voc_audio=idata[1]

    voc_audio=voc_audio
    return (np.real(voc_audio).astype(np.int16))


def play_stream(part_len_seconds):
    """
        Play back wav file
        :param filePath: String, the source file
        :return: None
        """

    global _ImRunning
    _ImRunning = True
    CHUNK=44100
    RATE=SAMPLE_RATE
    bits=bytes(np.ravel(input))
    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        #datap = process_chunk(in_data)
        int_data=np.fromstring(in_data, np.int16).astype(np.float32)
        print(int_data.max())
        float_data=int_data/32256

        datap=(vocoder(float_data))
        return (datap, pyaudio.paContinue)

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer = CHUNK,
                    stream_callback=callback
                    )
    # start the stream (4)
    stream.start_stream()


    while stream.is_active():
        time.sleep(part_len_seconds)
        stream.stop_stream()
        print("Stream is stopped")

    # stop stream (6)
    stream.stop_stream()
    stream.close()

    # close PyAudio (7)
    p.terminate()

def processLiveAudio(liveBuffer=None, drumkit=None, quant_factor=1.0, iters=0, method='NMFD', thresholdAdj=0., part_len_sec=2):
    """
    main logic for source separation, onset detection and tempo extraction and quantization
    :param liveBuffer: numpy array, the source audio
    :param drumkit: list of drums
    :param quant_factor: float, amount of quantization (change to boolean)
    :param iters: int, number of runs of nmfd for bagging separation
    :param method: The source separation method, 'NMF' or 'NMFD
    :param thresholdAdj: float, adjust the onset detection thresholds, one value for all drums.
    :return: list of drums containing onset locations in hits field and mean tempo of the take
    """

    onset_alg = 2
    #filt_spec = utils.stft(liveBuffer)
    stacks = 1
    total_priors = 0
    filtos=[]
    for i in range(len(drumkit)):
        total_priors += drumkit[i].get_heads().shape[2]
        total_priors += drumkit[i].get_tails().shape[2]
    Wpre = np.zeros((FILTERBANK_SHAPE, total_priors, utils.max_n_frames))
    total_heads = 0

    for i in range(len(drumkit)):
        heads = drumkit[i].get_heads()
        K1 = heads.shape[2]
        ind = total_heads
        for j in range(K1):
            Wpre[:, ind + j, :] = heads[:, :, j]
            total_heads += 1
    total_tails = 0

    for i in range(len(drumkit)):
        tails = drumkit[i].get_tails()
        K2 = tails.shape[2]
        ind = total_heads + total_tails
        for j in range(K2):
            Wpre[:, ind + j, :] = tails[:, :, j]
            total_tails += 1
        #Make a dummy history
        drumkit[i].set_peaks(np.zeros((32,)))
    def check_for_onsets(data, Wpre, total_heads, current_frame):
        #print(len(data))
        filt_spec=utils.stft(data,streaming=True)
        #print(filt_spec.shape)
        filtos.extend(filt_spec[3:])
        #print(filt_spec.shape)
        if method == 'NMFD' or method == 'ALL':
            H, Wpre, err1 = nmfd.NMFD(filt_spec.T, iters=128, Wpre=Wpre, include_priors=True, n_heads=total_heads, hand_break=True)
        if method == 'NMF' or method == 'ALL':
            H, err2 = nmfd.semi_adaptive_NMFB(filt_spec.T, Wpre=Wpre, iters=128, n_heads=total_heads, hand_break=True)

        onsets = np.zeros(H[0].shape[0])
        total_heads = 0

        for i in range(len(drumkit)):
            heads = drumkit[i].get_heads()
            K1 = heads.shape[2]
            ind = total_heads
            for k in range(K1):
                index = ind + k
                HN = H[index]
                HN = HN / HN.max()
                if k == 0:
                    H0 = HN
                else:
                    H0 += HN
                total_heads += 1
            if i == 0:
                onsets = H0
            else:
                onsets = onsets + H0
            #print(H0.shape)
            history=drumkit[i].get_peaks()
            H0=np.concatenate((history, H0))
            peaks = onset_detection.pick_onsets(H0, threshold=drumkit[i].get_threshold() + thresholdAdj)
            drumkit[i].set_peaks(H0[4:36])
            # remove extrahits used to level peak picking algorithm:
            #print(H0)
            peaks = peaks[np.where(peaks > history.shape[0])]
           # print(peaks)
            peaks = peaks[np.where(peaks < 36)]

            #
            #print(i, peaks)
            drumkit[i].set_hits(peaks)
        return drumkit, Q_HOP/HOP_SIZE
    #stream section:
    CHUNK =HOP_SIZE #hop size when int16
    RATE = SAMPLE_RATE
    BUFFER_SIZE=FRAME_SIZE
    global BUFFER_LOC
    BUFFER_LOC = 0
    p = pyaudio.PyAudio()
    BUF=np.zeros(BUFFER_SIZE, dtype=np.int16)
    def callback(in_data, frame_count, time_info, status):
        global BUFFER_LOC
        #grab globals
        if BUFFER_LOC<int(BUFFER_SIZE/CHUNK):
            BUF[(BUFFER_LOC*CHUNK):(BUFFER_LOC+1)*CHUNK]=np.fromstring(in_data, np.int16)
            BUFFER_LOC+=1
        elif BUFFER_LOC>=int(BUFFER_SIZE/CHUNK):
            data = np.concatenate((BUF[:-CHUNK], np.fromstring(in_data, np.int16)), axis=None)
            BUF[:-CHUNK]=data[CHUNK:]
            BUFFER_LOC += 1
            datap = check_for_onsets(data, Wpre, total_heads, BUFFER_LOC)
            drums=['kick','snare', 'chh', 'ohh']
            for i in range(len(datap[0])):
                if len(datap[0][i].get_hits())>0:
                    print (drums[i], datap[0][i].get_hits(), BUFFER_LOC)

        return (in_data, pyaudio.paContinue)

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback
                    )
    # start the stream (4)
    stream.start_stream()


    while stream.is_active():
        time.sleep(part_len_sec)
        stream.stop_stream()
        print("Stream is stopped")

    # stop stream (6)
    stream.stop_stream()
    stream.close()
    #plt.figure(figsize=(10, 6))
    #filtos=np.array(filtos)
    #print(filtos.shape)
    #plt.imshow((filtos.T),aspect='auto', origin='lower')
    #plt.show()
    # close PyAudio (7)
    p.terminate()



def debug():
    #game.initKitBG('./Kits/säng/', drumwise=True, method='NMF')
    #drum_kit = game.loadKit('./Kits/säng/')
    #print(drum_kit)
    #processLiveAudio(None,drum_kit, part_len_sec=50, method='NMF')
    play_stream(200)

if __name__ == "__main__":
    debug()
