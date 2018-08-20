from utils import *
import pandas as pd
import madmom
#
# fpr = np.zeros((proc.shape[1], nrOfDrums * 2 * K, ConvFrames))
# # frames=np.zeros((8192,nrOfDrums*nrOfPeaks))
# drums = []
# list_X = []
# list_y = []
# # NoneTemplates=[]
# highEmph = [0, 0, 0, 1, 0, 0, 1, 1, 0]
# # highEmph=[2,2,2,2,2,2,2,2,2]
# ###Tässä pitää napata talteen framet/sample
# for i in range(nrOfDrums):
#     try:
#         soundcheck = False
#         print("\rdrum{}.wav".format(i),end='', flush=True)
#
#         buffer = madmom.audio.Signal("{}drum{}.wav".format(DRUMKIT_PATH, i), frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
#         CC1, freqtemps, threshold = getPeaksFromBuffer(buffer, 1, nrOfPeaks, highEmph=highEmph[i])
#         for j in range(K):
#             ind = i * K
#             fpr[:, ind + j, :] = freqtemps[0][:, :, j]
#             fpr[:, ind + j + nrOfDrums * K, :] = freqtemps[1][:, :, j]
#
#     except Exception as e:
#         print(e)
#         print('samples not found, please soundcheck!')
#         print("Play drum nr. {}".format(i + 1))
#         CC1, freqtemps, threshold, buffer = getStompTemplate(nrOfPeaks, recordingLength=2, highEmph=highEmph[i])
#         # outBuffer=unFrameSignal(buffer)
#         madmom.io.audio.write_wave_file(buffer, './drum{}.wav'.format(i), sample_rate=SAMPLE_RATE)
#
#     if (True):
#         templates = []
#         samples = []
#
#         for j in range(len(CC1)):
#             t = CC1[j]
#
#             tinyBuff = make_sample(buffer, t, n_frames=4)
#             #templates.append(generate_features(tinyBuff, highEmph[i]))
#
#             samples.append(tinyBuff)
#
#         drums.append(
#             Drum(name=[i], highEmph=highEmph[i], peaks=CC1, templates=templates, samples=samples, threshold=threshold,
#                  midinote=midinotes[i], probability_threshold=1))
# print ("\nSamples loaded")

try:

    buffer = madmom.audio.Signal("{}drumBeatAnnod.wav".format(DRUMKIT_PATH), frame_size=FRAME_SIZE, hop_size=HOP_SIZE)


except Exception as e:
    print(e)
    print('jotain meni vikaan!')
showEnvelope(buffer)
showFFT(get_preprocessed_spectrogram(buffer).T)