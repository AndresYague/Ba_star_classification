import copy

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.mixture import GaussianMixture
from scipy.stats import kde
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy import odr
from matplotlib import ticker
from functools import reduce
import matplotlib
from matplotlib.ticker import FormatStrFormatter
#from plot_correl import *

# load in the variables defined in the main plotting file but used here
#tmp_file_variables = open("tmp_variables.txt", "w")
#solZrNb, SAVE_DIR = np.loadtxt("tmp_variables.txt", dtype="str")
#solZrNb = float(solZrNb)
#solZrNb = plot_correl.solZrNb
#SAVE_DIR = plot_correl.SAVE_DIR

peak1 = ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru']
#peak1 = ['Sr', 'Y', 'Zr', 'Nb', 'Mo'] # for resids vs resids 5 elems plot
# peak1 = ['Zr', 'Nb']
peak2 = ['La', 'Ce', 'Nd', 'Sm', 'Eu']

label_fontsize = 38
config = None

def set_config(cfg):
    global solZrNb, SAVE_DIR, onlyBa, Nb_lim
    solZrNb = cfg['solZrNb']
    SAVE_DIR = cfg['SAVE_DIR']
    onlyBa = cfg['onlyBa']
    Nb_lim = cfg['Nb_lim']

def create_mean(df, df_resid_obs):
    df['star'] = df_resid_obs.loc['star']
    for col in df.columns:
        if col != 'star':
            df[col] = pd.to_numeric(df[col])
    if 'Nb/Fe_res' in df.columns:
        #return df.groupby('star').mean().T
        df['Nb_res_avg'] = df.groupby('star')['Nb/Fe_res'].transform('mean').T
    elif 'Zr/Fe_res' in df.columns:
        #return df.groupby('star').mean().T
        df['Zr_res_avg'] = df.groupby('star')['Zr/Fe_res'].transform('mean').T
    return df.T


def separate_Fe_peaks(df_toselect, peak1, peak2):
    colname_peak1 = []
    colname_peak2 = []
    for colname in df_toselect.index:
        numer = colname[:colname.find('/')]
        denom = colname[colname.find('/')+1:]
        if denom in ['Fe', 'Fe_res', 'Fe_obs']:
            if numer in peak1:
                colname_peak1.append(colname)
            elif numer in peak2:
                colname_peak2.append(colname)

    df_peak1 = df_toselect.loc[colname_peak1]
    df_peak2 = df_toselect.loc[colname_peak2]
    return df_peak1, df_peak2

def unique(list1):
    # Print directly by using * symbol
    ans = reduce(lambda re, x: re + [x] if x not in re else re, list1, [])
    print(len(ans))

def plot_peak(xpeak_df, ypeak_df, xtype, ytype, peakname, nrow, ncol, mass , absFe=False, p2p1=False, cmass=False, triag2=False):
    resid_obs = False
    if xtype == "obs" and ytype == "res": resid_obs = True

    if Nb_lim >= 0: dil_lst = xpeak_df.loc['dil']
    if xtype == "res": xpeak_df = xpeak_df.T.filter(like='res').T
    if (xtype == "obs") and (ytype != "obs") : xpeak_df = xpeak_df.T.filter(like='obs').T
    if ((ytype == "res") and (xtype not in ["feh", "mass", "dil"])) or (Nb_lim >= 0):
        ypeak_df_res = ypeak_df.T.filter(like='res')
        ypeak_df_res.columns = ypeak_df_res.columns.str.replace('_res', '')
        ypeak_df_obs = ypeak_df.T.filter(like='obs')
        ypeak_df_obs.columns = ypeak_df_obs.columns.str.replace('_obs', '')
        if Nb_lim >= 0:
            ypeak_df = copy.deepcopy(ypeak_df_obs).T
            xpeak_df_res = xpeak_df.T.filter(like='res')
            xpeak_df_res.columns = xpeak_df_res.columns.str.replace('_res', '')
            xpeak_df_obs = xpeak_df.T.filter(like='obs')
            xpeak_df_obs.columns = xpeak_df_obs.columns.str.replace('_obs', '')
            xpeak_df = copy.deepcopy(xpeak_df_obs).T
            xpeak_df.loc['b'] = ypeak_df_obs['Zr/Fe'] - xpeak_df.loc['Nb/Fe']
            xpeak_df.loc['om'] = 10**(xpeak_df.loc['b']+solZrNb)

        #ypeak_df = ypeak_df_obs.subtract(ypeak_df_res, fill_value=np.nan).T
    if (ytype == "obs") and (xtype not in ["feh", "obs"]):
        ypeak_df = ypeak_df.T.filter(like='obs').T
        ypeak_df = ypeak_df.T.filter(like='/Fe').T

    if p2p1:
        nrow += 1
        ncol += 1

    if not triag2: fig, ax = plt.subplots(nrow-1, ncol-1, sharex='col', sharey=True, figsize=[ncol*5.3, (nrow-1)*8])
    else: fig, ax = plt.subplots(nrow-1, ncol-1, sharex='col', sharey=True, figsize=[ncol*5.3, (nrow-1)*8])
    if Nb_lim >= 0:
        size = [16,16]
        if Nb_lim == 0: size = [x*0.991 for x in size]
        fig, ax = plt.subplots(nrow-1, ncol-1, sharex='col', figsize=size)
    nsubplot = 0
    if ytype != "feh":
        ratios_x_lst = list(xpeak_df.index)

    ratios_y_lst = list(ypeak_df.index)
    print(ratios_y_lst)

    # Search for global max and min
    toplot_x_all, toplot_y_all = [], []
    for row in range(nrow-1):
        for col in range(ncol-1):
            if absFe and (row == 1) and col == len(peak2): break

            # if normal, col >= row subplots are used, if flipped, row >= col
            if triag2: grt, sml = row, col
            else: grt, sml = col, row
            # Set what to plot on y axis
            if (grt >= sml) or (p2p1 == True):
                toplot_x, toplot_y, _, ratioy = get_data_toplot(row, col, nsubplot, xtype, resid_obs, p2p1, ratios_x_lst, ratios_y_lst, xpeak_df, ypeak_df, triag2)
                if Nb_lim == 0:
                    toplot_x = xpeak_df_res['Nb/Fe']
                    toplot_y = xpeak_df.loc['om']
                toplot_x_arr = np.array(toplot_x)
                if xtype == "feh": toplot_x = toplot_x_arr[np.where(toplot_x_arr > -0.7)]

                toplot_y = list(toplot_y)
                sy = np.array(np.sort(np.array(toplot_y)))
                sy = sy[np.logical_not(np.isnan(sy))]

                for i in range(0,3):
                    if (abs(sy[-1]) - abs(sy[-4])) > 0.05:
                        sy = np.delete(sy, -1) # only one outlier

                toplot_x_all.extend(toplot_x)
                toplot_y_all.extend(sy)
                nsubplot += 1


    toplot_x_all = [float(i) for i in toplot_x_all]
    toplot_y_all = [float(i) for i in toplot_y_all]
    toplot_all = toplot_x_all + toplot_y_all
    xmin = np.nanmin(toplot_x_all)-0.05
    ymin = np.nanmin(toplot_y_all)-0.05
    xmax = np.nanmax(toplot_x_all)+0.05 # 0.35
    ymax = np.nanmax(toplot_y_all)+0.02
    ymax = ymax + (ymax-ymin)*0.1
    if Nb_lim == 0:
        ymin -= 1
        ymax += 1
        xmax += .1
        xmin -= .1
    if xtype == "dil": xmin = 0.001


    # FOR EACH SUBPLOT
    nsubplot = 0
    missing_plot = 0
    for row in range(nrow-1):
        for col in range(ncol-1):
            if nrow != 2:
                current_ax = ax[row, col-missing_plot]
            elif ncol != 2: # only one row
                current_ax = ax[col]
            else:
                current_ax = plt.gca()

            if absFe and (row == 1) and col >= len(peak2):
                break

            # Set what to plot on y axis
            if triag2: grt, sml = row, col
            else: grt, sml = col, row

            # Set what to plot on y axis
            if (grt >= sml) or (p2p1 == True):
                toplot_x, toplot_y, rationame_x, rationame = get_data_toplot(row, col, nsubplot, xtype, resid_obs, p2p1, ratios_x_lst, ratios_y_lst, xpeak_df, ypeak_df, triag2)
                if Nb_lim == 0:
                    toplot_x = xpeak_df_res['Nb/Fe']
                    toplot_y = xpeak_df.loc['om']
                if Nb_lim > 0:
                    ymax = 3
                    xmax = 3

                # mesh for KDE
                xi, yi = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
                arr = create_arr(toplot_x, toplot_y)
                index = np.isnan(arr).any(axis=0)
                arr = np.delete(arr, index, axis=1)

                kde_exists = True
                vals_x = copy.deepcopy(toplot_x.values)
                vals_y = copy.deepcopy(toplot_y.values)
                if vals_x.shape[0] > 1: vals_x = vals_x.T.astype(float)
                if vals_y.shape[0] > 1: vals_y = vals_y.T.astype(float)
                if not ((vals_x == vals_y) | (np.isnan(vals_x) & np.isnan(vals_y))).all():
                    kde_exists = True  # y-axis is same as x-axis, there is no kde
                else:
                    missing_plot = 1
                    nsubplot += 1
                    rationame_x = toplot_x.name
                    ax[0, len(peak2)-missing_plot].xaxis.set_tick_params(labelbottom=True, rotation=30, labelsize=label_fontsize - 6)
                    ax[0, len(peak2)-missing_plot].set_xlabel("[{:}]$_{:}$".format(rationame_x, "{obs}"), fontsize=label_fontsize)
                    continue

                if kde_exists:
                    k, zi = create_kde(arr, xi, yi)

                    zik = zi.reshape(xi.shape)
                    xarr = arr[0, :]
                    yarr = arr[1, :]

                    current_ax.axhline(0, color='black', ls='-', lw=2, alpha=.3)
                    current_ax.axvline(0, color="black", ls="-", lw=2, alpha=.3)
                    current_ax.yaxis.set_ticks_position('both')
                    current_ax.xaxis.set_ticks_position('both')
                    current_ax.xaxis.set_tick_params(direction="in", length=10)
                    current_ax.yaxis.set_tick_params(direction="in", length=10)

                if ytype == "res":
                    if cmass:
                        current_ax.scatter(toplot_x, toplot_y, c=mass, cmap='hot_r', s=22, edgecolors='black', linewidth=.4)
                    else:
                        if kde_exists:
                            pc = current_ax.pcolormesh(xi, yi, zi.reshape(xi.shape)/np.amax(zi), cmap=plt.cm.pink_r)
                        current_ax.scatter(xarr, yarr, c='black', s=20, edgecolors='beige', linewidth=1)
                else:
                    if cmass:
                        current_ax.scatter(toplot_x, toplot_y, c=mass, cmap='viridis', s=15, edgecolors='beige', linewidth=.8)
                    if Nb_lim >= 0:
                        if kde_exists:
                            if onlyBa:
                                #pc = current_ax.pcolormesh(xi, yi, zi.reshape(xi.shape)/np.amax(zi), cmap=plt.cm.PuBuGn)
                                pc = current_ax.pcolormesh(xi, yi, zi.reshape(xi.shape)/np.amax(zi), cmap=plt.cm.Greys)


                        carr = xpeak_df_res['Nb/Fe']
                        carrnan = carr.isnull()
                        carrnotnan = ~carrnan

                        def ind_agb(ind_Ba, ind_ini, dil):
                            ind_Ba = ind_Ba.astype(float)
                            #ind_ini = ind_ini.astype(float)
                            dil = dil.astype(float)
                            #return np.log10(dil**(-1) * (10**ind_Ba - (1-dil)*10**ind_ini))
                            ind_agb_lst = []
                            #for i in range(len(ind_Ba)):
                            #    ind_agb_lst.append(np.log10(dil[i].pow(-1))) #* (10**ind_Ba[i] - (1-dil[i])*10**ind_ini[i])))
                            #return np.log10(dil.pow(-1))
                            #return ind_agb_lst
                            result = []

                            for ba, dil_n in zip(ind_Ba, dil):
                                if ba is None or dil_n is None:
                                    result.append(None)
                                else:
                                    result.append(np.log10(dil_n ** (-1) * (10 ** ba - (1 - dil_n) * 10 ** ind_ini)))

                            return pd.Series(result)

                        toplot_x_agb = ind_agb(xpeak_df.loc['Nb/Fe'].T, 0.0, dil_lst)
                        toplot_y_agb = ind_agb(ypeak_df.loc['Zr/Fe'].T, 0.0, dil_lst)
                        #if Nb_lim == 0:
                        #    toplot_x_agb = toplot_x
                        #toplot_y_agb = toplot_y_agb - toplot_y
                        #toplot_x_agb = toplot_x
                        #toplot_x = toplot_x_agb
                        #toplot_y = toplot_y_agb
                        if not onlyBa:
                            xpeak_df.loc['b_AGB'] = toplot_y_agb - toplot_x_agb
                            xpeak_df.loc['om_AGB'] = 10 ** (xpeak_df.loc['b_AGB'] + solZrNb)
                        xpeak_df.loc['om'] = 10 ** (xpeak_df.loc['b'] + solZrNb)

                        if Nb_lim == 0:
                            toplot_x_agb = xpeak_df_res['Nb/Fe']
                            toplot_y_agb = xpeak_df.loc['om_AGB']

                        if onlyBa: arr = create_arr(toplot_x, toplot_y)
                        else: arr = create_arr(toplot_x_agb, toplot_y_agb)
                        index = np.isnan(arr).any(axis=0)
                        arr = np.delete(arr, index, axis=1)
                        xarr = arr[0, :]
                        yarr = arr[1, :]

                        xmin = min(toplot_x)-0.05
                        xmax = max(toplot_x_agb)+0.3
                        if onlyBa: xmax = max(toplot_x)+0.05
                        if Nb_lim == 0:
                            ymin = min(toplot_y_agb)-2
                            ymax = max(toplot_y_agb)+2
                        else:
                            ymin = min(toplot_y)-0.05
                            ymax = max(toplot_y_agb)+0.1
                        if onlyBa: ymax = max(toplot_y)+0.05

                        if onlyBa:
                            sc = current_ax.scatter(toplot_x[carrnotnan], toplot_y[carrnotnan], c=carr[carrnotnan], s=100, edgecolors='darkslategray', linewidth=1, cmap='hot', alpha=1, zorder=100)
                            current_ax.scatter(toplot_x[carrnan], toplot_y[carrnan], c='#00000042', s=100, edgecolors='darkslategray', linewidth=1.5, zorder=100)
                        else:
                            current_ax.scatter(toplot_x[carrnotnan], toplot_y[carrnotnan], c='gray', marker='x', s=140, edgecolors='darkslategray', linewidth=2.8, cmap='hot', alpha=1, zorder=100)
                            sc = current_ax.scatter(toplot_x_agb[carrnotnan], toplot_y_agb[carrnotnan], c=carr[carrnotnan], s=100, edgecolors='darkslategray', linewidth=1, cmap='hot', alpha=1, zorder=100)

                        #current_ax.scattefr(toplot_x[carr.isnull()], toplot_y[carr.isnull()], c=np.where(carrnan, 'magenta', carrnan), edgecolors='black', s=100)
                        #current_ax.scatter(toplot_x, toplot_y, c=list(xpeak_df_res['Nb/Fe']), s=42, edgecolors='beige', linewidth=1, cmap="inferno", alpha=1)
                        #plt.colorbar(sc, pad=0.1)
                        if Nb_lim > 0:
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("bottom", size="5%", pad=1.8)
                            plt.colorbar(sc, cax=cax, orientation="horizontal")
                            current_ax.locator_params(axis='both', nbins=6)
                            cax.locator_params(axis='x', nbins=10)
                            cax.xaxis.set_tick_params(labelbottom=True)
                            cax.set_xlabel(r"$\Delta$ [Nb/Fe]")


                    else:
                        if kde_exists:
                            pc = current_ax.pcolormesh(xi, yi, zi.reshape(xi.shape)/np.amax(zi), cmap=plt.cm.PuBuGn)
                        current_ax.scatter(toplot_x, toplot_y, c='black', s=20, edgecolors='beige', linewidth=1)


                popt, pcov = curve_fit(linfn, xarr, yarr)#, sigma=(1/np.sqrt(k(arr)**2)))
                def linfn_fix(x, b): return x+b
                popt2, pcov2 = curve_fit(linfn_fix, xarr, yarr)

                if (xtype not in ["feh", "dil"]) and (Nb_lim!=0):
                    print("ODR!")
                    odr_model = odr.Model(linfn_target)
                    # Create a Data object using sample data created.
                    data = odr.Data(xarr, yarr)
                    # Set ODR with the model and data.
                    ordinal_distance_reg = odr.ODR(data, odr_model, beta0=[*popt])

                    # Run the regression.
                    out = ordinal_distance_reg.run()

                    # print the results
                    popt = out.beta
                    perr = np.sqrt(np.diag(pcov))
                    #print(popt)

                if xtype not in ["feh", "dil"]:
                    print("ODR!")
                    odr_model = odr.Model(linfn_fix_target)
                    # Create a Data object using sample data created.
                    data = odr.Data(xarr, yarr)
                    # Set ODR with the model and data.
                    ordinal_distance_reg = odr.ODR(data, odr_model, beta0=[*popt2])

                    # Run the regression.
                    out = ordinal_distance_reg.run()

                    # print the results
                    popt2 = out.beta
                    perr = np.sqrt(np.diag(pcov))
                    perr_odr = out.sd_beta


                #perr = np.sqrt(np.diag(pcov))
                #correl = pearsonr(xarr, yarr)
                correl = spearmanr(xarr, yarr)

                # xtype_plot = xtype
                # if (xtype == "feh") and (toplot_x.name != "Fe/H"): # that is a fake fe/h plot!
                #     xtype_plot = toplot_x.name
                #     print(xtype_plot)
                rationame_x = toplot_x.name
                set_labels(current_ax, row, col, nrow, ncol, xtype, ytype, rationame_x, rationame, p2p1, resid_obs, absFe, triag2, label_fontsize)
                #xlabels = ["{:.1f}".format(x) for x in current_ax.get_xticks()]
                #current_ax.set_xticklabels(xlabels, rotation=-45)
                if absFe and (row == 0) and (col >= len(peak2)):
                    if Nb_lim >= 0: current_ax.xaxis.set_tick_params(labelbottom=True, labelsize=label_fontsize-6)
                    else: current_ax.xaxis.set_tick_params(labelbottom=True, rotation=30, labelsize=label_fontsize-6)
                    if xtype == "dil":
                        current_ax.set_xlabel(r"$\delta$", fontsize=label_fontsize)
                current_ax.locator_params(axis='x', nbins=5)
                current_ax.locator_params(axis='y', nbins=6)
                if Nb_lim >= 0: current_ax.locator_params(axis='both', nbins=7)

                nsubplot += 1

            #elif ([row, col] != [1, 0]):
            else: current_ax.remove() # so erase non-necessary subplots

            x = np.linspace(xmin, xmax, 10)
            indmax = np.where(zik== np.amax(zik))
            maxpoint_x = xi[indmax[0][0], 0]
            maxpoint_y = yi[0, indmax[1][0]]


            if ytype == "res":
                current_ax.plot(x, linfn(x, *popt), color='deepskyblue', ls='--', lw=5, label=r'$a =${:.2f}'.format(popt[0])) #deepskyblue
                #current_ax.scatter(x_tofit, y_tofit, color="deepskyblue", s=8)
                current_ax.plot(x, linfn(x, correl[0], maxpoint_y-correl[0]*maxpoint_x), color='royalblue', ls='', lw=5) #label=r'$r_S =${:.2f}'.format(correl[0]))
            else:
            #     current_ax.plot(x, linfn(x, *popt), color='crimson', ls='--', lw=3, label=r'$a =${:.2f}'.format(popt[0]))
                lab = r'$a =${:.2f}'.format(popt[0])
                c = 'orangered'
                if Nb_lim >= 0:
                    lab = r'$a =${:5.2f}, $\,b \,\,\,\,\,= ${:.2f}'.format(popt[0], popt[1])
                    c = 'lawngreen'
                    current_ax.plot(x, linfn(x, *popt), color='black', ls='-', lw=6, alpha=0.5, zorder=200)  # deepskyblue
                current_ax.plot(x, linfn(x, *popt), color=c, ls='--', lw=5, label="Line 1: "+lab, zorder=201)  # deepskyblue

                current_ax.plot(x, linfn(x, correl[0], maxpoint_y-correl[0]*maxpoint_x), color='yellow', ls='', lw=2) #label=r'$r_S =${:.2f}'.format(correl[0]))
                if Nb_lim > 0:
                    bs = xpeak_df.loc['b']
                    poptplot = np.nanmax(bs)
                    current_ax.plot(x, linfn_fix(x, poptplot), color='royalblue', ls='-.', lw=5,
                                    label=r'Line 2: $b =${:5.2f}, $\omega^* =${:.1f}'.format(poptplot, 10**(poptplot+solZrNb)))
                    current_ax.plot(x, linfn_fix(x, popt2[0] + perr_odr[0]), color='deepskyblue', ls='-', lw=5, zorder=400,
                                    label=r'Line 3: $b =${:5.2f}, $\omega^* =${:.1f}'.format(popt2[0], 10**(popt2[0]+solZrNb)))  # deepskyblue
                    poptplot = np.nanmin(bs)
                    #bs = np.array(bs).astype(float)
                    poptplot =  np.sort(bs)[2]
                    current_ax.plot(x, linfn_fix(x, poptplot), color='royalblue', ls='--', lw=5,
                                    label=r'Line 4: $ b =${:5.2f}, $\omega^* =${:.1f}'.format(poptplot, 10**(poptplot+solZrNb)))  # deepskyblue
                    poptplot = np.log10(16.6)-solZrNb
                    #current_ax.plot(x, linfn_fix(x, poptplot), color='b', ls='-', lw=3, alpha=0.6,
                    #                label=r'$b =${:5.2f}, $\omega^* =${:.1f}'.format(poptplot, 10**(poptplot+solZrNb)))
                    #current_ax.scatter([0.189, 0.291, 0.309, 0.371, 0.461, 0.421, 0.581, 0.772, 0.811], [0.286, 0.289, 0.350, 0.409, 0.590, 0.659, 0.779, 0.829, 0.850], color='b', marker='s', s=120, alpha=0.6)
                    poptplot = np.log10(11.5)-solZrNb
                    current_ax.plot(x, linfn_fix(x, poptplot), color='r', ls='-', lw=3, alpha=0.6,
                                    label=r'Line 5: $b =${:5.2f}, $\omega^* =${:.1f}'.format(poptplot, 10**(poptplot+solZrNb)))
                    current_ax.legend(fontsize=28, loc="lower right")
                    current_ax.yaxis.set_label_position("left")
                    xpeak_df.loc['om'].to_csv('intercept_om.dat', index=False)
                if Nb_lim == 0:
                    current_ax.set_ylabel(r'$\omega^*$')
                    current_ax.set_xlabel(r'$\Delta$ [Nb/Fe]')
                    current_ax.yaxis.set_label_position("left")

            #current_ax.plot(x, linfn(x, correl2[0], maxpoint_y - correl2[0] * maxpoint_x), color='yellow', ls=':')

            # 45° line through max of kde
            # max_xcoord = xi[np.where(zik == np.amax(zik))][0]
            # max_ycoord = yi[np.where(zik == np.amax(zik))][0]
            # if popt[0] > 0: current_ax.plot(x, x+max_ycoord-max_xcoord, color="crimson", ls="--")
            # else:
            #     current_ax.plot(x, x+max_ycoord-max_xcoord, color="crimson", ls="--")
            #     #current_ax.plot(x, -x+max_ycoord+max_xcoord, color="crimson", ls="--")

            # xmin=-0.3
            # xmax=0.3
            # ymin=-0.3
            # ymax=0.3
            #ax.set_box_aspect(3/4)
            #if Nb_lim > 0:
                #current_ax.set_aspect("equal", "box")
                #current_ax.set_xlim(0, 2.5)
            current_ax.set_xlim(xmin, xmax)
            current_ax.set_ylim(ymin, ymax)
            if Nb_lim >= 0: current_ax.xaxis.set_tick_params(rotation=0)
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            #current_ax.legend(loc='upper right', fontsize=26)
            #current_ax.text('$r_S$ = {:.2f},  $a$ = {:.2f}'.format(correl.statistic, popt[0]),
                            #xmax, ymax,
                            #transform=current_ax.transAxes,
                            #bbox=dict(boxstyle='round', fc='w'))

            if Nb_lim <= 0:
                if not triag2: fontsiz = label_fontsize-3
                else: fontsiz = label_fontsize-4
                current_ax.annotate('$r_S$ = {:.2f},  $a$ = {:.2f}'.format(correl.statistic, popt[0]), xy=(0.5, 1),
                            xytext=(0, -20), fontsize=fontsiz,
                            xycoords='axes fraction', textcoords='offset points',
                            bbox=dict(facecolor='white', boxstyle='round', alpha=1),
                            horizontalalignment='center', verticalalignment='top', zorder=10000)




    plt.tight_layout()
    if absFe:
        fig.subplots_adjust(wspace=0, hspace=0.15)
    else: fig.subplots_adjust(wspace=0, hspace=0)

    cbaxnum = [1,0]
    if triag2: cbaxnum = [0,1]
    if absFe: cbaxnum = [1,5]

    if absFe:
        pos = ax[cbaxnum[0], cbaxnum[1]].get_position()
        cax = fig.add_axes([pos.x0 + pos.width*0.1, pos.y0 + pos.height*0.64, pos.width, 0.42 /fig.get_figheight()])
        cb1 = plt.colorbar(pc, cax=cax , orientation="horizontal")
        cb1.ax.set_xlabel('KDE normalised', fontsize=32)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb1.locator = tick_locator
        cb1.update_ticks()
        cb1.ax.set_xticklabels(cb1.ax.get_xticklabels(), rotation=30)
        ax[cbaxnum[0], cbaxnum[1]].remove()
    elif Nb_lim < 0:
        #divider = make_axes_locatable(ax[nrow-2, ncol-2])
        #cax = divider.append_axes("right", size="5%", pad=1)
        #divider = make_axes_locatable(ax[-1, -1])
        pos_last = ax[-1, -1].get_position()
        cax = fig.add_axes([1.01, pos_last.y0, 0.42 /fig.get_figwidth(), pos_last.height])
        cbar = plt.colorbar(pc, cax=cax )
        cbar.ax.set_title('KDE \n normalised', pad=24, fontsize=32)

    if absFe: # erase unused subplots
        #for icol in [len(peak1)-2, len(peak1)-1]:
            #ax[1, icol].remove()
        ax[1, len(peak1)-1].remove()
        if missing_plot > 0:
            ax[1, len(peak1) - 2 - missing_plot].remove()


    prefix = "{:}_{:}".format(ytype, xtype)
    save_dir = "{:}/{:}".format(SAVE_DIR, prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('{:}/correl_{:}-{:}'.format(save_dir, prefix, peakname), bbox_inches='tight')

    #plt.show()



def get_data_toplot(row, col, nsubplot, xtype, resid_obs, p2p1, ratios_x_lst, ratios_y_lst, xpeak_df, ypeak_df, triag2):
    if xtype in ["feh", "mass", "dil"]:
        rationame = ratios_y_lst[nsubplot]
    else:
        rationame = ratios_y_lst[row]
    if (xtype !='feh') and triag2: rationame = ratios_y_lst[row+1]

    if resid_obs:
        rationame = rationame
    toplot_y = ypeak_df.loc[rationame]

    # Set what to plot on x axis
    if xtype == "feh":
        toplot_x = xpeak_df  # which is FeH in fact
        rationame_x = "Fe/H"
    elif '/' in xtype:
        rationame_x = xtype
    else:
        if p2p1 or (triag2 and (xtype!='feh')):
            rationame_x = ratios_x_lst[col]
        else:
            rationame_x = ratios_x_lst[col + 1]
        #if resid_obs:
        if (xtype == "feh") or resid_obs:
            rationame_x = rationame_x[:-4] + '_obs'
        if xtype == "mass": rationame_x = "mass"
        elif xtype == "dil": rationame_x = "dil"
        toplot_x = xpeak_df.loc[rationame_x]
    print(rationame)
    print(rationame_x)
    print()

    return toplot_x, toplot_y, rationame_x, rationame

def create_arr(toplot_x, toplot_y):
    return np.array(list(zip(list(toplot_x), list(toplot_y)))).T


def linfn(x, a, b):
    return a * x + b
    #return (x-b)/a

def linfn_target(p, x):
    a, b = p
    return a*x+b

def linfn_fix_target(p,x):
    b = p
    return x+b


def create_kde(arr, xi, yi):
    k = kde.gaussian_kde(arr, bw_method='scott')
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    return k, zi


def gaussmix(toplot_x, toplot_y, zik, current_ax):
    models, labels, aics = [], [], []
    data_formix = np.vstack((toplot_x, toplot_y))
    data_formix = data_formix[:, ~np.isnan(data_formix).any(axis=0)].T
    z_formix = k(data_formix.T)
    data_formix = data_formix[z_formix > np.amax(zik)*0.68]


    for numcomp in [1, 2]:
        mix_now = GaussianMixture(numcomp).fit(data_formix)
        models.append(mix_now)
        labels.append(mix_now.predict(data_formix))
        aic_now = mix_now.aic(data_formix)
        aics.append(aic_now)
        print("AIC: {:}".format(aic_now))

    aics = np.array(aics)
    ind = np.where(aics == min(aics))
    current_ax.scatter(data_formix[:,0], data_formix[:,1], c=labels[int(ind[0])])




def set_labels(current_ax, row, col, nrow, ncol, xtype, ytype, rationame_x, rationame, p2p1, resid_obs, absFe, triag2, label_fontsize):
    if rationame_x is not None:
        if ("res" in rationame_x) or ("obs" in rationame_x):
            rationame_x = rationame_x[:-4]
    if "obs" in rationame: rationame = rationame[:-4]
    if row == 0 or (triag2 and (row==col)):  # Sub-titles (if xtype is feh, what is the ratio)
        if xtype == "feh":
            if triag2: titletext = rationame[:rationame.find("/")]
            else: titletext = rationame[rationame.find("/") + 1:]
            if not triag2:
                if ytype == "res":
                    current_ax.set_title(r"$\Delta [$$A$/{:}]".format(titletext), fontsize=label_fontsize)
                else:
                    current_ax.set_title(r"[$A$/{:}]$_{:}$".format(titletext, "{obs}"), fontsize=label_fontsize)
            if triag2:
                if ytype == "res":
                    current_ax.set_title(r"$\Delta [${:}/$A$]".format(titletext), fontsize=label_fontsize)
                else:
                    current_ax.set_title(r"[{:}/$A$]$_{:}$".format(titletext, "{obs}"), fontsize=label_fontsize)

    if absFe:
        if ytype == "res": current_ax.set_title(r"$\Delta [${:}]".format(rationame))
        else: current_ax.set_title(r"[{:}]$_{:}$".format(rationame, "{obs}"))


    # XLABELS
    if ((p2p1 == False) and (col == row)) or (triag2 and row==nrow-2): # If triangular plot, the last row in each column
        # current_ax.set_ylabel(r"$\Delta$ [{:}/$A$]".format(rationame[:rationame.find("/")]), fontsize=label_fontsize)
        current_ax.xaxis.set_tick_params(labelbottom=True, direction="in", rotation=30, labelsize=label_fontsize-6)
        if not triag2: current_ax.yaxis.set_tick_params(labelbottom=True, direction="in")
        #current_ax.set_xlabel(r"[Fe/H]$_{:}$".format("{obs}"), fontsize=label_fontsize)  # loc='left'
        current_ax.set_xlabel(r"[{:}]$_{:}$".format(rationame_x, "{obs}"), fontsize=label_fontsize)  # loc='left'
        if xtype not in ["feh", "mass", "dil"]:
            if ytype == "res":
                current_ax.set_xlabel(r"$\Delta [${:}]".format(rationame_x, fontsize=label_fontsize))
                if resid_obs:
                    current_ax.set_xlabel(r"[{:}]$_{:}$".format(rationame_x, "{obs}"), fontsize=label_fontsize)
            else:
                current_ax.set_xlabel("[{:}]$_{:}$".format(rationame_x, "{obs}"), fontsize=label_fontsize)

    elif (p2p1 == True) and (row == nrow - 2): # If rectangular plot, the last row
        current_ax.xaxis.set_tick_params(labelbottom=True, rotation=30, labelsize=label_fontsize-6)
        # current_ax.set_xlabel("[Fe/H]", fontsize=label_fontsize) #loc='left'
        if xtype == "feh":
            current_ax.set_xlabel("[{:}]$_{:}$".format(rationame_x, "{obs}"), fontsize=label_fontsize)
            if ytype == "res":
                # current_ax.set_xlabel(r"$\Delta$ [{:}]".format(rationame_x), fontsize=label_fontsize)
                if resid_obs:
                    current_ax.set_xlabel(r"[{:}]$_{:}$".format(rationame_x, '{obs}'), fontsize=label_fontsize)

        elif xtype != "res":
            current_ax.set_xlabel(r"[{:}]$_{:}$".format(rationame_x, "{obs}"), fontsize=label_fontsize)
        elif xtype == "dil":
            current_ax.set_xlabel(r"$\delta$", fontsize=label_fontsize)
        else:
            current_ax.set_xlabel(r"$\Delta$[{:}]".format(rationame_x), fontsize=label_fontsize)

        if xtype == "dil": current_ax.set_xlabel(r"$\delta$", fontsize=label_fontsize)

    if absFe and (row == 0) and (col >= len(peak2)): # If rectangular plot and first row exceeds Peak 2 length (in the case of abundances vs Fe/H)
        current_ax.set_xlabel("[{:}]$_{:}$".format(rationame_x, "{obs}"), fontsize=label_fontsize)

    # YLABELS
    if (not triag2 and (col == ncol - 2)) or (triag2 and (col==0)):
        if not absFe:
            if xtype == "feh":
                if not triag2:
                    ylabel = rationame[:rationame.find("/")]
                    if ytype == "res":
                        current_ax.set_ylabel(r"$\Delta [${:}/$A$]".format(ylabel), fontsize=label_fontsize)
                    else:
                        current_ax.set_ylabel(r"[{:}/$A$]$_{:}$".format(ylabel, "{obs}"), fontsize=label_fontsize)

                else:
                    ylabel = rationame[rationame.find("/") + 1:]
                    if ytype == "res":
                        ylabel = ylabel[:ylabel.find("_")]
                        current_ax.set_ylabel(r"$\Delta [A/${:}]".format(ylabel), fontsize=label_fontsize)
                    else:
                        current_ax.set_ylabel(r"[$A$/{:}]$_{:}$".format(ylabel, "{obs}"), fontsize=label_fontsize)


            else:
                if ytype == "res":
                    current_ax.set_ylabel(r"$\Delta [${:}]".format(rationame[:-4]), fontsize=label_fontsize)
                    if resid_obs:
                        current_ax.set_ylabel(r"$\Delta [${:}]".format(rationame[:-4]), fontsize=label_fontsize)

                else:
                    current_ax.set_ylabel(r"[{:}]$_{:}$".format(rationame, "{obs}"), fontsize=label_fontsize)

            if not triag2: current_ax.yaxis.set_label_position("right")



