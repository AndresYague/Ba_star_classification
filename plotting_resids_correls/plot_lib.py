import pandas as pd
import numpy as np

def peakfilter(rowstar, peak1, peak2, res_obs_mode=False):
    '''Create the dataframes filtered by peaks and ordered by increasing atomic number'''

    cols = rowstar.index.values.tolist()
    peak1p1, peak1p2, peak2p1, peak2p2 = [pd.DataFrame() for i in range(4)]#[pd.Series(dtype='float64') for i in range(4)]
    p1p1_cols, p1p2_cols, p2p1_cols, p2p2_cols = [[] for i in range(4)]
    for rationame in cols:
        separate_pos = rationame.find("/")
        numerator = rationame[:separate_pos]
        denominator = rationame[separate_pos+1:]
        if numerator in peak1 and denominator in peak1: p1p1_cols.append(rationame)
        elif numerator in peak1 and denominator in peak2: p1p2_cols.append(rationame)
        elif numerator in peak2 and denominator in peak2: p2p2_cols.append(rationame)

        if res_obs_mode:
            if numerator in peak1 and denominator in [x+'_res' for x in peak1]:
                p1p1_cols.append(rationame)
            elif numerator in peak1 and denominator in [x+'_res' for x in peak2]:
                p1p2_cols.append(rationame)
            elif numerator in peak2 and denominator in [x+'_res' for x in peak2]:
                p2p2_cols.append(rationame)

        # now, change the denominator and numerator
        denominator = rationame[:separate_pos]
        numerator = rationame[separate_pos+1:]
        if numerator in peak2 and denominator in peak1:
            rname = str(numerator)+'/'+str(denominator)

    peak1p1 = rowstar.loc[p1p1_cols]
    peak1p2 = rowstar.loc[p1p2_cols]
    peak2p2 = rowstar.loc[p2p2_cols]


    peak2p1 = reverse_df(rowstar, peak1, peak2, res_obs_mode, only_reorder=False)
    peak2p2_reorder = reverse_df(rowstar, peak2, peak2, res_obs_mode, only_reorder=True)

    return peak1p1, peak1p2, peak2p1, peak2p2, peak2p2_reorder


def reverse_df(rowstar, was_first, was_second, res_obs_mode, only_reorder=False):
    '''Reverse the order of elements of peaks e.g. P1/P2 -> P2/P1'''

    cols = rowstar.index.values.tolist()
    direct_cols = []
    rev_cols = []

    for currel2 in was_second:
        for rationame in cols:
            separate_pos = rationame.find("/")
            numerator = rationame[:separate_pos]
            denominator = rationame[separate_pos + 1:]
            if (res_obs_mode and (numerator in was_first and (denominator == currel2+"_res"))) or (res_obs_mode == False and (numerator in was_first and (denominator == currel2))):
                if only_reorder: rname = str(numerator) + '/' + str(denominator)
                else: rname = str(denominator) + '/' + str(numerator)
                direct_cols.append(rname)
                rev_cols.append(rationame)

    rev_df = rowstar.loc[rev_cols]#.multiply(-1)
    for key,val in rev_df.items():
        tomult = rev_df[key]
        if not only_reorder:
            if isinstance(tomult, list):
                rev_df[key] = list(np.array(tomult)*(-1))
            else:
                rev_df[key] = tomult*(-1)
    rev_df.index = direct_cols
    return rev_df