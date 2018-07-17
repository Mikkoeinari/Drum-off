from utils import *
import pandas as pd
import madmom
from time import time
t0=time()
#live audio
print("Souncheck")

fpr = np.zeros((proc.shape[1], nrOfDrums * 2 * K, ConvFrames))
# frames=np.zeros((8192,nrOfDrums*nrOfPeaks))
drums = []
list_X = []
list_y = []
# NoneTemplates=[]
highEmph = [0, 0, 0, 1, 0, 0, 1, 1, 0]
# highEmph=[2,2,2,2,2,2,2,2,2]
###Tässä pitää napata talteen framet/sample
for i in range(nrOfDrums):
    try:
        soundcheck = False
        print("\rdrum{}.wav".format(i),end='', flush=True)

        buffer = madmom.audio.Signal("{}drum{}.wav".format(DRUMKIT_PATH, i), frame_size=2048, hop_size=HOP_SIZE)
        CC1, freqtemps, threshold = getPeaksFromBuffer(buffer, 1, nrOfPeaks, highEmph=highEmph[i])
        for j in range(K):
            ind = i * K
            fpr[:, ind + j, :] = freqtemps[0][:, :, j]
            fpr[:, ind + j + nrOfDrums * K, :] = freqtemps[1][:, :, j]

    except Exception as e:
        print(e)
        print('samples not found, please soundcheck!')
        print("Play drum nr. {}".format(i + 1))
        CC1, freqtemps, threshold, buffer = getStompTemplate(nrOfPeaks, recordingLength=2, highEmph=highEmph[i])
        # outBuffer=unFrameSignal(buffer)
        madmom.io.audio.write_wave_file(buffer, './drum{}.wav'.format(i), sample_rate=44100)

    if (True):
        templates = []
        samples = []

        for j in range(len(CC1)):
            t = CC1[j]

            tinyBuff = make_sample(buffer, t, n_frames=4)
            #templates.append(generate_features(tinyBuff, highEmph[i]))

            samples.append(tinyBuff)

        drums.append(
            Drum(name=[i], highEmph=highEmph[i], peaks=CC1, templates=templates, samples=samples, threshold=threshold,
                 midinote=midinotes[i], probability_threshold=1))
print ("\nSamples loaded")

# peakList = []
# for i in drums[:nrOfDrums]:
#     for k in range(K):
#         peakList.append(detector(i, hitlist=None))
try:

    buffer = madmom.audio.Signal("{}drumBeatAnnod.wav".format(DRUMKIT_PATH), frame_size=2048, hop_size=HOP_SIZE)


except Exception as e:
    print(e)
    print('jotain meni vikaan!')
plst = processLiveAudio(liveBuffer=buffer, peakList=drums, basis=fpr, quant_factor=0.0)

#print f-score:
print('\n\n')
hits = pd.read_csv("{}midiBeatAnnod.csv".format(DRUMKIT_PATH), sep="\t", header=None)
precision, recall, fscore, true_tot=0,0,0,0
for i in plst:
   predHits=frame_to_time(i.get_hits())

   #print(predHits, predHits.shape[0] )
   actHits=hits[hits[1]==i.get_name()[0]]
   actHits = actHits.iloc[:,0]
   #print(actHits.values, actHits.shape[0])
   trueHits=k_in_n(actHits.values,predHits, window=0.02)
   #print(trueHits)

   prec, rec, f_drum=f_score(trueHits, predHits.shape[0], actHits.shape[0])
   print(prec)
   print(rec)
   print(f_drum)
   print(trueHits)
   print('\n')
   #Multiply by n. of hits to get real f-score in the end.
   precision+=prec*actHits.shape[0]
   recall+=rec*actHits.shape[0]
   fscore+=(f_drum*actHits.shape[0])
   true_tot+=actHits.shape[0]
   #add_to_samples_and_dictionary(i.drum, buffer, i.get_hits())
print('Precision: {}'.format(precision/true_tot))
print('Recall: {}'.format(recall/true_tot))
print('F-score: {}'.format(fscore/true_tot))

'''
todo: Normalize freq -bands locally to adjust to signal level changing during performance
    frame by frame or something else, a window of fixeld length maybe?

'''

times = []
for i in plst:
    hits = frame_to_time(i.get_hits())
    labels = np.full(len(hits), i.get_midinote(), np.int64)
    inst = zip(hits, labels)
    times.extend(inst)
times.sort()
df = pd.DataFrame(times, columns=['time', 'inst'])
df['duration'] = pd.Series(np.full((len(times)), 0, np.int64))
df['vel'] = pd.Series(np.full((len(times)), 127, np.int64))
df = df[df.time != 0]
print('done!')
madmom.io.midi.write_midi(df.values, 'midi_testit_.mid')
df.to_csv('testbeat.csv', index=False, header=False, sep="\t")

print('Processing time:%0.2f' % (time()-t0))

# Thresholdit checkissä.


