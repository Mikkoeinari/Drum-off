import pandas as pd
from utils import *
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

midi_to_simple('./kakkosnelonen.csv')
simple_to_binary('./kakkosnelonen.csv')