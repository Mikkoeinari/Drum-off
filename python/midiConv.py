import pandas as pd
from utils import *
import pickle
def simple_to_binary(filename):
    d = pd.read_csv(filename, header=None, sep="\t").as_matrix()
    d1=time_to_frame(d[:, 0])
    d2=d[:,1].astype(int)

    data=mergerowsandencode(list(zip(d1,d2)))
    bindf=pd.DataFrame(data, columns=['inst'])
    bindf.to_csv(filename, index=True, header=False, sep="\t")

def midi_to_simple(filename):
    midinotes = [36, 38, 42, 46, 50, 45, 51, 49, 40]  # BD, SN, CHH, OHH, TT, FT, RD, CR, SHH
    markup = pd.read_csv(filename, names=['inst', 'time'], sep="\t", usecols=[2, 5])
    #print(markup.head(20))
    latencyCompensation = 20
    # +markup['time'].iloc[0]-markup['time'].iloc[1])
    #print(latencyCompensation)
    markup = markup.iloc[:]
    bdmask = (markup['inst'] == 36) | (markup['inst'] == 86)
    snmask = (markup['inst'] == 38) | (markup['inst'] == 40) | (markup['inst'] == 87) | (markup['inst'] == 85)
    chhmask = (markup['inst'] == 79) | (markup['inst'] == 46) | (markup['inst'] == 42)
    ohhmask = (markup['inst'] == 78)
    shhmask = (markup['inst'] == 44)
    ttmask = (markup['inst'] == 48) | (markup['inst'] == 14)
    ft1mask = (markup['inst'] == 47) | (markup['inst'] == 18)
    ridemask = (markup['inst'] == 51) | (markup['inst'] == 52) | (markup['inst'] == 53)
    crmask = (markup['inst'] == 16) | (markup['inst'] == 17) | (markup['inst'] == 57) | (markup['inst'] == 55) | (
                markup['inst'] == 59) | (markup['inst'] == 49)
    crapmask = (markup['inst'] == 23) | (markup['inst'] == 19) | (markup['inst'] == 37)
    markup = markup.set_value(bdmask, 'inst', 0)
    markup = markup.set_value(snmask, 'inst', 1)
    markup = markup.set_value(chhmask, 'inst', 2)
    markup = markup.set_value(ohhmask, 'inst', 3)
    markup = markup.set_value(ttmask, 'inst', 4)
    markup = markup.set_value(ft1mask, 'inst', 5)
    markup = markup.set_value(ridemask, 'inst', 6)
    markup = markup.set_value(crmask, 'inst', 7)
    markup = markup.set_value(shhmask, 'inst', 8)
    markup = markup.set_value(crapmask, 'inst', 9)
    # markup=markup.set_value(bdsnchhmask, 'inst', 8)
    # jne.
    markup['time'] = markup['time'].apply(lambda x: (x + latencyCompensation) / 1000.0)
    markup = markup[['time', 'inst']]
    #print(markup.head(20))
    markup.to_csv(filename, index=False, header=False, sep="\t")

#midi_to_simple('./kakkosnelonen.csv')
#simple_to_binary('./kakkosnelonen.csv')

from midi import read_midifile
from midi.events import *
import os
import pathlib
midis = []#[f for f in os.listdir('../midis/') if not f.startswith('.')]
for path, subdirs, files in os.walk('../midis'):
    for name in files:
        file=str(pathlib.PurePath(path, name))
        midis.append(file)


def quadBar(pattern):
    """
    makes one bar of pattern four bars
    :param pattern: list, pattern to multiply
    :return: np.array four bars of pattern
    """
    newpat=np.zeros_like((pattern*4))
    pattern=np.array(pattern)
    maxTick=pattern[-1][0]
    for i in range(4):
        for j in range(pattern.shape[0]):
            newpat[j+(i*pattern.shape[0])]=[pattern[j][0]+maxTick*i+1,pattern[j][1],pattern[j][2]]
    return newpat.tolist()

def extend_midi(old, new):
    maxtime=old[-1][0]
    for i in range(len(new)):
        new[i][0]=new[i][0]+maxtime
    return old+new

def format_data(data):
    #data = data[data['time'] <= 10000000]

    data['time']=time_to_frame((data['time']).astype(float)*.0052,sr=44100, hop_length=2**9).astype(int)
    bd = [35, 36]
    mask = data.pitch.isin(bd)
    data.loc[mask, 'pitch'] = 0
    sn=[37,38,39,40,54,73,74,75,76,77,78,79]
    mask = data.pitch.isin(sn)
    data.loc[mask, 'pitch'] = 1
    chh=[42,69,70,81]
    mask = data.pitch.isin(chh)
    data.loc[mask, 'pitch'] = 2
    ohh=[46,58]
    mask = data.pitch.isin(ohh)
    data.loc[mask, 'pitch'] = 3
    shh=[44,80]
    mask = data.pitch.isin(shh)
    data.loc[mask, 'pitch'] = 8
    tt=[48,50,60,62,63,65]
    mask = data.pitch.isin(tt)
    data.loc[mask, 'pitch'] = 4
    ft=[41,43,45,47,61,64,63,66,67,68]
    mask = data.pitch.isin(ft)
    data.loc[mask, 'pitch'] = 5
    rd=[51,53,56,59,71,72]
    mask = data.pitch.isin(rd)
    data.loc[mask, 'pitch'] = 6
    cr=[49,52,55,57]
    mask = data.pitch.isin(cr)
    data.loc[mask, 'pitch'] = 7
    data['vel']=100
    data=data[data.pitch.notnull()]
    data=data[data['pitch']<=9]
    data_hits=mergerowsandencode(data[['time','pitch']].as_matrix())
    data=pd.DataFrame(data_hits)
    return data
    #data.to_csv('midi_data_set/dataklimp{}b.csv'.format(i), index=True, header=None, sep='\t')
    #print(data.head())

masterfile=[]
filenro=0
lap=10000
for file in midis:
    if len(masterfile)>lap:
        lap+=10000
        print(len(masterfile))
    if len(masterfile) > 100000:
        masterfile=np.array(masterfile)
        data=pd.DataFrame(masterfile,columns=['time', 'pitch', 'velocity'])
        data=format_data(data)
        print(data.head())
        data.to_csv('midi_data_set/mididata{}.csv'.format(filenro), index=True, header=None, sep='\t')
        filenro+=1
        lap=10000
        masterfile=[]
        #break
    if not file.lower().endswith('.mid'):
        print(pathlib.Path(file).suffix)
        continue
    try:
        midi =read_midifile(file)
    except Exception as e:
        print(file, 'produced', e)
    midi.make_ticks_abs()
    tracks = [track for track in midi]
    eventCount = 0
    pattern=[]
    for track in tracks:
        for event_number, event in enumerate(track, eventCount):
            if isinstance(event, NoteOnEvent):
                if event.get_velocity()>0:
                    pattern.append([event.tick,event.get_pitch(),event.get_velocity()])
            if isinstance(event, EndOfTrackEvent):
                pattern.append([event.tick,None, None])
    if len(masterfile)>0:
        masterfile=extend_midi(masterfile,quadBar(pattern))
    else:
        masterfile=quadBar(pattern)
#for i in range(0,1):
 #   data=pd.read_csv('dataklimp{}.csv'.format(i), index_col=0,names=['time','pitch', 'vel'],header=None, sep='\t')


