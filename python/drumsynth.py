#import pygame, sys
#from pygame.locals import *
import pandas as pd
import numpy as np
import utils
import time
import pyaudio
import wave

t0 = time.time()
_ImRunning = False
bd=None
def open_and_merge(filename):
    global bd
    x=wave.open(filename, 'rb')
    bd=x
    x = x.readframes(x.getnframes())
    return x
bdBytes = open_and_merge('./Sounds/bigkit/mono/bd.wav')
sdBytes = open_and_merge('./Sounds/bigkit/mono/sn.wav')
chhBytes = open_and_merge('./Sounds/bigkit/mono/chh.wav')
ohhBytes = open_and_merge('./Sounds/bigkit/mono/ohh.wav')
ttBytes = open_and_merge('./Sounds/bigkit/mono/tt.wav')
ftBytes = open_and_merge('./Sounds/bigkit/mono/ft.wav')
rdBytes = open_and_merge('./Sounds/bigkit/mono/rd.wav')
crBytes = open_and_merge('./Sounds/bigkit/mono/cr1.wav')
shhBytes = open_and_merge('./Sounds/bigkit/mono/shh.wav')
sounds = [bdBytes, sdBytes, chhBytes, ohhBytes, ttBytes, ftBytes, rdBytes, crBytes, shhBytes]

CHUNK = 2048
buffer = np.zeros(2048)


def frame_to_time(frames, sr=bd.getframerate()):
    """
    Transforms frame numbers to time values
    :param frames: list of integers to transform
    :param sr: int, Sample rate of the FFT
    :param hop_length: int, Hop length of FFT
    :param hops_per_frame: ??
    :return: Numpy array of time values
    """
    # samples = (np.asanyarray(frames) * (hop_length / hops_per_frame)).astype(int)
    return frames / float(sr)


def playFile(filePath):
    global wf
    global _ImRunning
    _imRunning = True
    d = pd.read_csv(filePath, header=None, sep="\t").values
    d = list(utils.truncZeros(np.array(d[:, 1])))
    d = utils.splitrowsanddecode(d)
    gen = pd.DataFrame(d, columns=['time', 'inst'])
    gen['time'] = utils.frame_to_time(gen['time'], hop_length=int(utils.Q_HOP*2))
    d = gen.values
    #result wave
    wauva = wave.open('./tulos.wav', 'w')
    #pitääkö tsekata sampleja lukiessa???
    wauva.setparams((bd.getnchannels(), bd.getsampwidth(),bd.getframerate(), 0, 'NONE', 'not compressed'))
    #file pointer
    cursor = 0
    #max file size
    fileLength = int((d[-1][0] * 44100  / 8)+ len(sounds[int(d[-1][1])]))
    outfile = np.zeros((fileLength,), dtype=int)
    outfile = bytearray(outfile.tobytes())
    for i in range(len(d)):
        drum = int(d[i][1])
        # cursor=cursor+sounds[drum]
        if i != 0:
            gap = d[i][0] - d[i - 1][0]
            # print(drum)
            # print(np.rint(gap*44100))
            gapLen = int(np.rint(gap * 44100))
            cursor = cursor + gapLen
            pause = np.zeros((gapLen,), dtype=np.int16)
            # wauva.writeframesraw(pause.tobytes())
        # wauva.writeframesraw(sounds[drum])
        endCursor = cursor + (len(sounds[drum]))
        data1 = np.fromstring(bytes(outfile[cursor:endCursor]), np.int16)
        data2 = np.fromstring(sounds[drum], np.int16)
        outfile[cursor:endCursor] = ((data1 * .5 + data2 * .5)).astype(np.int16).tostring()
    #fix result size to 10s.
    outfile=outfile[:int(441000*4)]
        # wf=sounds[drum]
    wauva.writeframes(outfile)
    wauva.close()
    wauva = wave.open('tulos.wav', 'rb')
    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        data = wauva.readframes(frame_count)
        return (data, pyaudio.paContinue)

    stream = p.open(format=p.get_format_from_width(bd.getsampwidth()),
                    channels=1,
                    rate=bd.getframerate(),
                    output=True,
                    stream_callback=callback
                    )
    # start the stream (4)
    stream.start_stream()

    # wait for stream to finish (5)
    while stream.is_active() and _imRunning:
        time.sleep(0.1)

    # stop stream (6)
    stream.stop_stream()

    stream.close()

    wauva.close()

    # close PyAudio (7)
    p.terminate()

    # stream = p.open(format=p.get_format_from_width(bd.getsampwidth()),
    #                 channels=1,
    #                 rate=bd.getframerate(),
    #                 output=True,
    #
    #                 )
    # def atten(a):
    #     a=np.fromstring(a, np.int16)
    #     a=a//3
    #     return a.tobytes()
    # stream.start_stream()
    # feed=wauva
    #
    # data=feed.readframes(CHUNK)
    #
    # while len(data)>0:
    #
    #     stream.write(data, exception_on_underflow=False)
    #
    #     data = feed.readframes(CHUNK)

    # stream.stop_stream()
    # stream.close()
    # p.terminate()
    # wauva.close()
    # return True

#Uncommment for debugging
#playFile('./Kits/Mikon/takes/testbeat1535439315.332094.csv')
