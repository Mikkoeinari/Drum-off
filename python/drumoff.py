from utils import *
from time import time
import threading

t0 = time()
import pickle

# live audio
#print("Souncheck")

# fpr = np.zeros((proc.shape[1], nrOfDrums * 2 * K, ConvFrames))
# drums = []
list_X = []
list_y = []
msg = ''
highEmph = [0, 0, 0, 1, 0, 0, 1, 1, 0]


###
def soundcheckDrum(drumkit_path, drumId):
    CC1, freqtemps, threshold, buffer = getStompTemplate(nrOfPeaks)
    #print('jessus')
    madmom.io.audio.write_wave_file(buffer, '{}/drum{}.wav'.format(drumkit_path, drumId), sample_rate=SAMPLE_RATE)
    # try:
    #
    #     print("\rdrum{}.wav".format(drumId))
    #
    #     buffer = madmom.audio.Signal("{}/drum{}.wav".format(drumkit_path, drumId), frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    #     CC1, freqtemps, threshold = getPeaksFromBuffer(buffer, 1, nrOfPeaks, highEmph=highEmph[drumId])
    #     for j in range(K):
    #         ind = drumId * K
    #         fpr[:, ind + j, :] = freqtemps[0][:, :, j]
    #         fpr[:, ind + j + nrOfDrums * K, :] = freqtemps[1][:, :, j]
    # except Exception as e:
    #     print(e)
    #     #yield ('samples not found, please soundcheck!')
    #     #yield ("Play drum nr. {}".format(drumId + 1))
    #     CC1, freqtemps, threshold, buffer = getStompTemplate(nrOfPeaks, recordingLength=2, highEmph=highEmph[drumId])
    #     madmom.io.audio.write_wave_file(buffer, '{}/drum{}.wav'.format(drumkit_path, drumId), sample_rate=SAMPLE_RATE)


def initKit(drumkit_path, nrOfDrums, k=0):
    K=k
    k=10
    global drums, fpr
    #print(nrOfDrums)
    fpr = np.zeros((proc.shape[1], nrOfDrums * 2 * K, k))
    drums = []
    for i in range(nrOfDrums):

        buffer = madmom.audio.Signal("{}/drum{}.wav".format(drumkit_path, i), frame_size=FRAME_SIZE,
                                     hop_size=HOP_SIZE)
        CC1, freqtemps, threshold = getPeaksFromBuffer(buffer, 1, nrOfPeaks, k, K)
        for j in range(K):
            ind = i * K
            fpr[:, ind + j, :] = freqtemps[0][:, :, j]
            fpr[:, ind + j + nrOfDrums * K, :] = freqtemps[1][:, :, j]

        if (True):
            templates = []
            samples = []

            for j in range(len(CC1)):
                t = CC1[j]

                tinyBuff = make_sample(buffer, t, n_frames=4)
                # templates.append(generate_features(tinyBuff, highEmph[i]))

                samples.append(tinyBuff)

            drums.append(
                Drum(name=[i], highEmph=highEmph[i], peaks=CC1, templates=templates, samples=samples,
                     threshold=threshold,
                     midinote=midinotes[i], probability_threshold=1))
    # Pickle the important data
    pickle.dump(drums, open("{}/pickledDrumkit.drm".format(drumkit_path), 'wb'))
    pickle.dump(fpr, open("{}/pickledFpr.drm".format(drumkit_path), 'wb'))


def loadKit(drumkit_path):
    """
    Loads drumkit data to memory
    :param drumkit_path: path to drumkit
    :return: None
    """
    global drums, fpr
    drums = pickle.load(open("{}/pickledDrumkit.drm".format(drumkit_path), 'rb'))
    fpr = pickle.load(open("{}/pickledFpr.drm".format(drumkit_path), 'rb'))


def playLive(drumkit_path):
    ###T live input
    try:
        buffer = liveTake()
    except Exception as e:
        print(e)

    t0 = time()
    plst = processLiveAudio(liveBuffer=buffer, peakList=drums, Wpre=fpr, quant_factor=1.0)
    print('NMFDtime:%0.2f' % (time() - t0))
    times = []
    bintimes = []
    for i in plst:
        hits = i.get_hits()
        binhits = i.get_hits()

        hits = frame_to_time(hits)
        ##TVASTA QUANTIZE?????
        labels = np.full(len(hits), i.get_midinote(), np.int64)
        binlabels = np.full(len(binhits), i.get_name(), np.int64)
        inst = zip(hits, labels)
        bininst = zip(binhits, binlabels)
        times.extend(inst)
        bintimes.extend(bininst)
    times.sort()
    bintimes.sort()
    bintimes = mergerowsandencode(bintimes)
    df = pd.DataFrame(times, columns=['time', 'inst'])
    df['duration'] = pd.Series(np.full((len(times)), 0, np.int64))
    df['vel'] = pd.Series(np.full((len(times)), 127, np.int64))
    bindf = pd.DataFrame(bintimes, columns=['inst'])
    fileName='{}/takes/testbeat{}.csv'.format(drumkit_path, time())
    bindf.to_csv(fileName, index=True, header=False, sep="\t")
    df = df[df.time != 0]
    print('done!')
    CreateMidi = False
    if CreateMidi:
        madmom.io.midi.write_midi(df.values, 'midi_testit_.mid')
        generated = splitrowsanddecode(bintimes)
        gen = pd.DataFrame(generated, columns=['time', 'inst'])
        gen.to_csv('generated_enc_dec0.csv', index=False, header=None, sep='\t')
        print('pattern generating time:%0.2f' % (time() - t0))
        # change to time and midinotes
        gen['time'] = frame_to_time(gen['time'])
        gen['inst'] = to_midinote(gen['inst'])
        gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
        gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))
        madmom.io.midi.write_midi(gen.values, 'midi_testit_enc_dec0.mid')
    return fileName
    print('Processing time:%0.2f' % (time() - t0))


def soundCheck(drumkit_path, nrOfDrums, drumkit_drums):
    for i in range(nrOfDrums):
        try:
            soundcheck = False
            yield ("\rdrum{}.wav".format(i))

            buffer = madmom.audio.Signal("{}drum{}.wav".format(DRUMKIT_PATH, i), frame_size=FRAME_SIZE,
                                         hop_size=HOP_SIZE)
            CC1, freqtemps, threshold = getPeaksFromBuffer(buffer, 1, nrOfPeaks, highEmph=highEmph[i])
            for j in range(K):
                ind = i * K
                fpr[:, ind + j, :] = freqtemps[0][:, :, j]
                fpr[:, ind + j + nrOfDrums * K, :] = freqtemps[1][:, :, j]

        except Exception as e:
            yield (e)
            yield ('samples not found, please soundcheck!')
            yield ("Play drum nr. {}".format(i + 1))

            def runThreaded():
                CC1, freqtemps, threshold, buffer = getStompTemplate(nrOfPeaks, recordingLength=2, highEmph=highEmph[i])

            threading.Thread(target=runThreaded()).start()
            # outBuffer=unFrameSignal(buffer)
            madmom.io.audio.write_wave_file(buffer, '{}/drum{}.wav'.format(drumkit_path, i), sample_rate=SAMPLE_RATE)

        if (True):
            templates = []
            samples = []

            for j in range(len(CC1)):
                t = CC1[j]

                tinyBuff = make_sample(buffer, t, n_frames=4)
                # templates.append(generate_features(tinyBuff, highEmph[i]))

                samples.append(tinyBuff)

            drums.append(
                Drum(name=[i], highEmph=highEmph[i], peaks=CC1, templates=templates, samples=samples,
                     threshold=threshold,
                     midinote=midinotes[i], probability_threshold=1))
        yield ('cont')
    print("\nSamples loaded")
    yield 'done'


# peakList = []
# for i in drums[:nrOfDrums]:
#     for k in range(K):
#         peakList.append(detector(i, hitlist=None))
def play(filePath):
    try:

        buffer = madmom.audio.Signal("{}drumBeatAnnod.wav".format(filePath), frame_size=FRAME_SIZE,
                                     hop_size=HOP_SIZE)


    except Exception as e:
        print(e)
        print('jotain meni vikaan!')
    t0=time()
    fs=np.zeros((7,9))
    for n in range(1,7):
        initKit('../trainSamplet/', 9, k=n)
        plst = processLiveAudio(liveBuffer=buffer, peakList=drums, Wpre=fpr, quant_factor=0.0, iters=86, method='NMFD', K=n)
        print('\nNMFDtime:%0.2f' % (time() - t0))
        annotated = True
        if (annotated):
            # print f-score:
            print('\n\n')
            hits = pd.read_csv("{}midiBeatAnnod.csv".format(filePath), sep="\t", header=None)
            precision, recall, fscore, true_tot = 0, 0, 0, 0
            for i in plst:
                predHits = frame_to_time(i.get_hits())

                # print(predHits, predHits.shape[0] )
                actHits = hits[hits[1] == i.get_name()[0]]
                actHits = actHits.iloc[:, 0]
                # print(actHits.values, actHits.shape[0])
                trueHits = k_in_n(actHits.values, predHits, window=0.02)
                # print(trueHits)2
                prec, rec, f_drum = f_score(trueHits, predHits.shape[0], actHits.shape[0])
                print(prec)
                print(rec)
                print(f_drum)
                fs[n,i.get_name()]=f_drum
                print(trueHits)
                print('\n')
                # Multiply by n. of hits to get real f-score in the end.
                precision += prec * actHits.shape[0]
                recall += rec * actHits.shape[0]
                fscore += (f_drum * actHits.shape[0])
                true_tot += actHits.shape[0]
                # add_to_samples_and_dictionary(i.drum, buffer, i.get_hits())
            print('Precision: {}'.format(precision / true_tot))
            #fs[n,0]=(precision / true_tot)
            print('Recall: {}'.format(recall / true_tot))
            #fs[n,1] = (recall / true_tot)
            print('F-score: {}'.format(fscore / true_tot))
            #fs[n,2]=(fscore / true_tot)

    showEnvelope(fs[1:n], ('Kick','Snare','HH Closed','HH Open','Rack Tom', 'Floor Tom', 'Ride', 'Crash', 'HH Pedal'), ('templates/drum','score'))
    '''
    todo: Normalize freq -bands locally to adjust to signal level changing during performance
        frame by frame or something else, a window of fixeld length maybe?
    
    '''

    times = []
    bintimes = []
    for i in plst:
        hits = i.get_hits()
        binhits = i.get_hits()
        hits = frame_to_time(hits)
        labels = np.full(len(hits), i.get_midinote(), np.int64)
        binlabels = np.full(len(binhits), i.get_name(), np.int64)
        inst = zip(hits, labels)
        bininst = zip(binhits, binlabels)
        times.extend(inst)
        bintimes.extend(bininst)
    times.sort()
    bintimes.sort()
    bintimes = mergerowsandencode(bintimes)
    df = pd.DataFrame(times, columns=['time', 'inst'])
    df['duration'] = pd.Series(np.full((len(times)), 0, np.int64))
    df['vel'] = pd.Series(np.full((len(times)), 127, np.int64))
    bindf = pd.DataFrame(bintimes, columns=['inst'])
    bindf.to_csv('testbeat0.csv', index=True, header=False, sep="\t")
    df = df[df.time != 0]
    print('done!')

    madmom.io.midi.write_midi(df.values, 'midi_testit_.mid')
    generated = splitrowsanddecode(bintimes)
    gen = pd.DataFrame(generated, columns=['time', 'inst'])
    gen.to_csv('generated_enc_dec0.csv', index=False, header=None, sep='\t')
    print('pattern generating time:%0.2f' % (time() - t0))
    # change to time and midinotes
    gen['time'] = frame_to_time(gen['time'])
    gen['inst'] = to_midinote(gen['inst'])
    gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
    gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))
    madmom.io.midi.write_midi(gen.values, 'midi_testit_enc_dec0.mid')

    print('Processing time:%0.2f' % (time() - t0))
    return True
#debug
#initKit('../trainSamplet/',9)
loadKit('../trainSamplet/')
play('../trainSamplet/')
