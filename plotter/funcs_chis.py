import numpy as np
import pandas as pd

def chi_calc(df_dec_res, chi_start, chis, chiok_arr, modelnum):
    intv_start = df_dec_res.loc[chi_start].index
    df_dec_speak = df_dec_res[(df_dec_res.index >= chi_start) & (df_dec_res.index <= 63)][df_dec_res['deC_abund'].notna()]
    chis.append(0)
    num_element, element_count = 0, 0
    currres_arr = []
    for ll in range(len(df_dec_speak['res_dil'])):
        #currres = df_dec_speak['res_ori'].iloc[ll]
        currres = df_dec_speak['res_dil'].iloc[ll]
        currerr = df_dec_speak['error'].iloc[ll]
        if currerr == 0: currerr = 0.004 # to not divide by 0
        if chiok_arr[ll] == 1:
            if np.isnan(currerr) == False:
                num_element += 1
                chis[modelnum] += currres ** 2 / currerr ** 2
            else:
                currres_arr.append(currres**2)
                element_count += 1

    if num_element == 0: chis[modelnum] = np.nansum(currres_arr) / (0.2**2 *element_count) # error is considered 0.2 then
    else: chis[modelnum] = chis[modelnum] / num_element


def chi_rank(chis):
    chis_sorted = sorted(chis)
    chis_rank = []
    for kk in range(len(chis)):
        chis_rank.append(chis_sorted.index(chis[kk]))
    return chis_rank


def chi_fruity2(nplot_fr, fr_mass, dil_fr, chiok_arr, chi_start, df_fr, df_dec_toplot, fr_label_arr):
    modelnum = 0
    fr_plotarr, df_dec_res_arr = [], []

    for zind in range(nplot_fr):
        #currkey = 'fr' + str(fr_mass[zind])
        currkey = df_fr[zind].keys()[1+zind]
        dil_now = dil_fr[zind]
        dd = 1 - dil_now + 10 ** (df_fr[zind][currkey])*dil_now
        fr_plotarr.append(np.log10(dd))
        # fr_plotarr.append(df_fr[zind][currkey])

        # residual
        df_dec_res = pd.merge(df_dec_toplot, df_fr[zind][currkey], on='Z', how='inner')
        df_dec_res['res_dil'] = df_dec_toplot['deC_abund'] - fr_plotarr[modelnum]
        df_dec_res['res_ori'] = df_dec_toplot['deC_abund'] - df_fr[zind][currkey]
        df_dec_res['ori_mod'] = df_fr[zind][currkey]
        df_dec_res['dil_err'] = dil_err_calc(df_fr[zind][currkey], df_fr[zind][currkey]*0.15, dil_now, dd)

        df_dec_speak = df_dec_res[(df_dec_res.index >= chi_start) & (df_dec_res.index <= 63)][df_dec_res['deC_abund'].notna()]
        for elem in range(len(df_dec_speak['res_dil'])):
            if (df_dec_speak['element'].iloc[elem] == 'Ba' or df_dec_speak['element'].iloc[elem] ==  'Rb'
                or df_dec_speak['element'].iloc[elem] ==  'Eu'):
                if abs(df_dec_speak['res_dil'].iloc[elem]) < 0.3:
                    chiok_arr[elem] = 1
            else: chiok_arr[elem] = 1

        df_dec_res_arr.append(df_dec_res)
        modelnum += 1

    return chiok_arr, fr_plotarr, df_dec_res_arr, modelnum
    

def chi_res_mon2(mon_mass_matching, mon_mmixes, dil_mon, chiok_arr, chi_start, df_moni, df_monf, df_dec_toplot, mon_label_arr):
    modelnum = 0 # how many models have been plotted
    mon_plotarr, df_dec_res_arr = [], []
    for kk in range(len(mon_mass_matching)):
        curr_col_mon = str(mon_mass_matching[kk]) + 'mix' + str(mon_mmixes[kk])  # current column name
        dil_now = dil_mon[kk]
        dd = 10 ** (df_moni[kk][curr_col_mon]) * (1 - dil_now) + 10 ** (df_monf[kk][curr_col_mon]) * dil_now
        mon_plotarr.append(np.log10(dd))

        #residual
        #df_dec_toplot = pd.merge(df_dec_toplot, df_monf[kk][curr_col_mon], on='Z', how='inner')
        df_dec_res = pd.merge(df_dec_toplot, mon_plotarr[modelnum], on='Z', how='inner')
        df_dec_res['res_dil'] = df_dec_toplot['deC_abund'] - mon_plotarr[modelnum] # subtracts on matching Z values!
        df_dec_res['res_ori'] = df_dec_toplot['deC_abund'] - df_monf[kk][curr_col_mon]
        df_dec_res['ori_mod'] = df_monf[kk][curr_col_mon]
        #df_dec_res['dil_err'] = dil_err_calc(df_monf[kk][curr_col_mon], df_monf[kk][curr_col_mon]*0.15, dil_now, dd)

        df_dec_speak = df_dec_res[(df_dec_res.index >= chi_start) & (df_dec_res.index <= 63)][df_dec_res['deC_abund'].notna()]
        for elem in range(len(df_dec_speak['res_dil'])):
            if (df_dec_speak['element'].iloc[elem] == 'Ba' or df_dec_speak['element'].iloc[elem] == 'Rb'
                or df_dec_speak['element'].iloc[elem] ==  'Eu'):
                if abs(df_dec_speak['res_dil'].iloc[elem]) < 0.3:
                    chiok_arr[elem] = 1
            else:
                chiok_arr[elem] = 1

        df_dec_res_arr.append(df_dec_res)
        modelnum += 1

    return chiok_arr, mon_plotarr, df_dec_res_arr, modelnum

def chis_okayed(fr_mon, countarr, df_dec_res_arr, chi_start, chiok_arr, fr_label_arr):
    modelnum = 0
    chis = []
    for zind in range(len(countarr)):
        for mind in range(countarr[zind]):
            chi_calc(df_dec_res_arr[modelnum], chi_start, chis, chiok_arr, modelnum)
            #if fr_mon == 'f': fr_label_arr[zind][mind] += r", {:3}{:<4.2f}".format(r"$\chi^2_{\nu}=\,$", chis[modelnum])	#chi2 kiiras
            #if fr_mon == 'm': fr_label_arr[modelnum] += r", {:3}{:<4.2f}".format(r"$\chi^2_{\nu}=\,$", chis[modelnum])		#chi2 kiiras
            modelnum += 1
    return chis

def dil_err_calc(Xf, Xf_err, dil, dd):
    return abs(10**Xf * dil / dd * Xf_err)

def fruity_plot(starn, df_ok, bestchi_label, ax1, ax2, nplot_fr, df_fr, df_dec_toplot, df_fr_res_arr, fr_plotarr, fr_label_arr, chis_rank, nplotted_all, res_file, color_map, mark):
    nplot_tot = sum(nplot_fr)
    coloridx_fr = color_map(np.linspace(0.1, 1, num=nplot_tot + 1))  # colormap distribution for Fruity
    n_plotted = 0
    for zind in range(len(nplot_fr)):
        for mind in range(nplot_fr[zind]):
            ax1.plot(df_fr[zind].index, fr_plotarr[n_plotted], color=coloridx_fr[n_plotted], linestyle='-', linewidth=3,
                     label=fr_label_arr[zind][mind], #+ " - {:^3.0f}".format(chis_rank[nplotted_all]+1)		# chi2 kiiras
                      zorder=3, marker=mark, alpha=0.8, markersize=10) 
            ax1.plot(df_fr[zind].index, fr_plotarr[n_plotted].interpolate(), color=coloridx_fr[n_plotted], linestyle='-',
                     linewidth=3, zorder=3, marker='', alpha=0.8, markersize=10)  # interpolates between missing data
            ax2.plot(df_fr_res_arr[n_plotted].index, df_fr_res_arr[n_plotted]['res_dil'], color=coloridx_fr[n_plotted], marker=mark,
                     linestyle='', markersize=10, markeredgewidth=1, zorder=3, alpha=0.8)
            #if chis_rank[nplotted_all] == 0: print(df_fr_res_arr[n_plotted]['res_dil'].sort_index())
            if chis_rank[nplotted_all] == 0:
                df_fr_res_arr[n_plotted]['ok'] = df_fr_res_arr[n_plotted]['dil_err'] + df_dec_toplot['error'].fillna(0.15) - abs(df_fr_res_arr[n_plotted]['res_dil'])
                #df_fr_res_arr[n_plotted].sort_index().loc[37:63].to_csv(res_file, sep=',', mode='a')
                #res_file.write("\n\n")
                df_ok['ok1'] =  df_fr_res_arr[n_plotted]['dil_err'] + df_dec_toplot['error'].fillna(0.15) - abs(df_fr_res_arr[n_plotted]['res_dil'])
                bestchi_label[0] = (fr_label_arr[zind][mind]+',')
            elif chis_rank[nplotted_all] == 1:
                df_fr_res_arr[n_plotted]['ok'] = df_fr_res_arr[n_plotted]['dil_err'] + df_dec_toplot['error'].fillna(
                    0.15) - abs(df_fr_res_arr[n_plotted]['res_dil'])
                df_ok['ok2'] = df_fr_res_arr[n_plotted]['dil_err'] + df_dec_toplot['error'].fillna(0.15) - abs(
                    df_fr_res_arr[n_plotted]['res_dil'])
                bestchi_label[1] = (fr_label_arr[zind][mind]+',')
            elif chis_rank[nplotted_all] == 2:
                df_fr_res_arr[n_plotted]['ok'] = df_fr_res_arr[n_plotted]['dil_err'] + df_dec_toplot['error'].fillna(
                    0.15) - abs(df_fr_res_arr[n_plotted]['res_dil'])
                df_ok['ok3'] = df_fr_res_arr[n_plotted]['dil_err'] + df_dec_toplot['error'].fillna(0.15) - abs(
                    df_fr_res_arr[n_plotted]['res_dil'])
                bestchi_label[2] = (fr_label_arr[zind][mind] + ',')
            n_plotted += 1
            nplotted_all += 1
    return nplotted_all
