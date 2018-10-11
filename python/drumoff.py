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
highEmph = [0, 0, 1, 1, 0, 0, 1, 1, 0]
K=1

###
def soundcheckDrum(drumkit_path, drumId):
    buffer = getStompTemplate(nrOfPeaks, convFrames=10)
    #print('jessus')
    madmom.io.audio.write_wave_file(buffer, '{}/drum{}.wav'.format(drumkit_path, drumId), sample_rate=SAMPLE_RATE)

def initKitBG(drumkit_path, nrOfDrums, K=1, rm_win=4, bs_len=32):
    K=K
    L=10
    global drums, fpr
    #print(nrOfDrums)
    drums = []
    drs = [2, 4, 6, 3, 2, 2, 5, 1, 8]
    #n_frames = [12, 9, 11, 17, 15, 13, 10, 11, 15]
    n_frames=np.full(9,10)
    filt_spec_all = 0
    shifts = []
    for i in range(nrOfDrums):
        L=n_frames[i]

        buffer = madmom.audio.Signal("{}/drum{}.wav".format(drumkit_path, i), frame_size=FRAME_SIZE,
                                     hop_size=HOP_SIZE)
        filt_spec = get_preprocessed_spectrogram(buffer, sm_win=4)
        peaks= getPeaksFromBuffer(filt_spec, 10, nrOfPeaks, L, K)
        freqtemps=findDefBinsBG(peaks, filt_spec, L, K, bs_len=bs_len)
        if i==0:
            filt_spec_all=filt_spec
            shifts=[0]
        else:
            shift=filt_spec_all.shape[0]
            filt_spec_all=np.vstack((filt_spec_all,filt_spec))
            shifts.append(shift)

        drums.append(
                Drum(name=[i], highEmph=highEmph[i], peaks=peaks, heads=freqtemps[0], tails=freqtemps[1],
                     threshold=0,
                     midinote=midinotes[i], probability_threshold=1))
    #prior picture
    # showFFT([drums[0].get_heads()[:,:,0],drums[1].get_heads()[:,:,0],drums[2].get_heads()[:,:,0],
    #         drums[3].get_heads()[:, :, 0], drums[4].get_heads()[:, :, 0], drums[5].get_heads()[:, :, 0],
    #         drums[6].get_heads()[:, :, 0], drums[7].get_heads()[:, :, 0], drums[8].get_heads()[:, :, 0]],)
    #showFFT([drums[0].get_heads()[:, :, 0], drums[1].get_heads()[:, :, 0], drums[2].get_heads()[:, :, 0]])
    recalculate_thresholds(filt_spec_all, shifts, drums, drumwise=False, rm_win=rm_win)
    # Pickle the important data
    pickle.dump(drums, open("{}/pickledDrumkit.drm".format(drumkit_path), 'wb'))
    #pickle.dump(fpr, open("{}/pickledFpr.drm".format(drumkit_path), 'wb'))

def initKit(drumkit_path, nrOfDrums, K=1, bs_len=32):
    K=K
    L=10
    global drums, fpr
    #print(nrOfDrums)
    fpr = np.zeros((proc.shape[1], nrOfDrums * 2 * K, L))
    drums = []
    filt_spec_all = 0
    shifts = []
    for i in range(nrOfDrums):

        buffer = madmom.audio.Signal("{}/drum{}.wav".format(drumkit_path, i), frame_size=FRAME_SIZE,
                                     hop_size=HOP_SIZE)
        #CC1, freqtemps, threshold = getPeaksFromBuffer(buffer, 1, nrOfPeaks, k, K)
        filt_spec = get_preprocessed_spectrogram(buffer, sm_win=4)
        peaks = getPeaksFromBuffer(filt_spec, 10, nrOfPeaks, L, K)
        freqtemps = findDefBins(peaks, filt_spec, L, K)
        #for j in range(K):
        #    ind = i * K
        #    fpr[:, ind + j, :] = freqtemps[0][:, :, j]
        #    fpr[:, ind + j + nrOfDrums * K, :] = freqtemps[1][:, :, j]
        if i==0:
            filt_spec_all=filt_spec
            shifts=[0]
        else:
            shift=filt_spec_all.shape[0]
            filt_spec_all=np.vstack((filt_spec_all,filt_spec))
            shifts.append(shift)

        drums.append(
            Drum(name=[i], highEmph=highEmph[i], peaks=peaks, heads=freqtemps[0], tails=freqtemps[1],
                 threshold=.15,
                 midinote=midinotes[i], probability_threshold=1))
    recalculate_thresholds(filt_spec_all, shifts, drums, drumwise=False)
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
    #fpr = pickle.load(open("{}/pickledFpr.drm".format(drumkit_path), 'rb'))


def playLive(drumkit_path):
    ###T live input
    try:
        buffer = liveTake()
    except Exception as e:
        print('liveplay:',e)

    t0 = time()
    plst = processLiveAudio(liveBuffer=buffer, drums=drums, quant_factor=0.0, iters=87, method='NMFD')
    print('NMFDtime:%0.2f' % (time() - t0))
    times = []
    bintimes = []
    for i in plst:

        hits = i.get_hits()
        if not len(hits):
            continue
        binhits = i.get_hits()
        hits = frame_to_time(hits)
        ##TVASTA QUANTIZE?????
        labels = np.full(len(hits), i.get_midinote(), np.int64)
        binlabels = np.full(len(binhits), i.get_name(), np.int64)
        inst = zip(hits, labels)
        bininst = zip(binhits, binlabels)
        times.extend(inst)
        bintimes.extend(bininst)
    if not len(times):
        return False
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


# drums = []
# for i in drums[:nrOfDrums]:
#     for k in range(K):
#         drums.append(detector(i, hitlist=None))
def play(filePath, K):
    try:

        buffer = madmom.audio.Signal("{}drumBeatAnnod.wav".format(filePath), frame_size=FRAME_SIZE,
                                     hop_size=HOP_SIZE)


    except Exception as e:
        print(e)
        print('jotain meni vikaan!')

    fs=np.zeros((17,3))

    for n in [1]:
        #print(2**n)

        #initKitBG(filePath, 9, K=K)#, rm_win=n, bs_len=350)
        t0 = time()
        plst = processLiveAudio(liveBuffer=buffer, drums=drums, quant_factor=0.0, iters=128, method='NMFD', rm_win=n)
        print('\nNMFDtime:%0.2f' % (time() - t0))
        #Print scores if annotated
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
                #fs[n,i.get_name()]=f_drum
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
    #showEnvelope(fs, ('Precision', 'Recall', 'f-score'), ('Max K','score'))

    #showEnvelope(fs[:n], ('Kick','Snare','HH Closed','HH Open','Rack Tom', 'Floor Tom', 'Ride', 'Crash', 'HH Pedal'), ('templates/drum','score'))
    '''
    todo: Normalize freq -bands locally to adjust to signal level changing during performance
        frame by frame or something else, a window of fixeld length maybe?
    
    '''

    times = []
    bintimes = []
    for i in plst:
        hits = i.get_hits()
        binhits = i.get_hits()
        hits = frame_to_time(hits, hop_length=Q_HOP)
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
    #print('pattern generating time:%0.2f' % (time() - t0))
    # change to time and midinotes
    gen['time'] = frame_to_time(gen['time'], hop_length=Q_HOP)
    gen['inst'] = to_midinote(gen['inst'])
    gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
    gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))
    from madmom.io import midi
    notes=madmom.utils.expand_notes(gen.values, duration=0.6, velocity=127)
    filu =midi.MIDIFile.from_notes(notes, tempo=DEFAULT_TEMPO)
    filu.save('midi_testit_enc_dec0.mid')
    #madmom.io.midi.write_midi(gen.values, 'midi_testit_enc_dec0.mid')
    #print('Processing time:%0.2f' % (time() - t0))
    return True

#Test method to check Eric Battenbergs onset detection function.
def testOnsDet(filePath, alg=0):
    buffer=0
    try:
        buffer = madmom.audio.Signal("{}drumBeatAnnod.wav".format(filePath), frame_size=FRAME_SIZE,
                                     hop_size=HOP_SIZE)
    except Exception as e:
        print(e)
        print('jotain meni vikaan!')
    if alg == 0:
        filtspec=get_preprocessed_spectrogram(buffer, sm_win=8, test=True)
        H0=filtspec[:,0]
    else:
        H0 = superflux(spec_x=filtspec.T, win_size=4)
    H0 = H0 / H0[3:].max()
    showEnvelope(H0[3:])
    peaks=pick_onsets(H0,delta=0.015)


    hits = pd.read_csv("{}midiBeatAnnod.csv".format(filePath), sep="\t", header=None)
    precision, recall, fscore, true_tot = 0, 0, 0, 0

    predHits = frame_to_time(peaks)
    actHits = hits[0]

    #actHits = actHits.iloc[:, 0]
    # print(actHits.values, actHits.shape[0])
    trueHits = k_in_n(actHits.values, predHits, window=0.058)
    # print(trueHits)2
    prec, rec, f_drum = f_score(trueHits, predHits.shape[0], actHits.shape[0])
    print(prec)
    print(rec)
    print(f_drum)
    # fs[n,i.get_name()]=f_drum
    print(trueHits)
    print('\n')
    # Multiply by n. of hits to get real f-score in the end.
    precision += prec * actHits.shape[0]
    recall += rec * actHits.shape[0]
    fscore += (f_drum * actHits.shape[0])
    true_tot += actHits.shape[0]
    # add_to_samples_and_dictionary(i.drum, buffer, i.get_hits())
    print('Precision: {}'.format(precision / true_tot))
    # fs[n,0]=(precision / true_tot)
    print('Recall: {}'.format(recall / true_tot))
    # fs[n,1] = (recall / true_tot)
    print('F-score: {}'.format(fscore / true_tot))
    # fs[n,2]=(fscore / true_tot)
#debug
#initKitBG('Kits/mcd2/',8,K)
#K=1
#initKitBG('../trainSamplet/',9,K=K,rm_win=6)
loadKit('../trainSamplet/')
play('../trainSamplet/', K=K)
#initKitBG('../DXSamplet/',9,K=K,rm_win=6)
#loadKit('../trainSamplet/')
#testOnsDet('../trainSamplet/', alg=0)
#play('../trainSamplet/', K=K)
#from math import factorial
#def comb(n, k):
#    return factorial(n) / factorial(k) / factorial(n - k)
#hitsize=[]
#for r in range(2,25):
#    hitsize.append(2*(comb(r,2))+32)
#showEnvelope(hitsize)