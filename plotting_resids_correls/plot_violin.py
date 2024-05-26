import sys, os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import seaborn as sns

import numpy as np
import pandas as pd
import seaborn as sns
# from process_data_lib import *
from processplot_data_lib import *
from plot_lib import *

allthree = False # True if CM, RF and NN models are plotted on each other for comparison; False if just one classifier
FIGNAME="viol-all3" # root of figure names
#sys.path.insert(0, "/correl1")

font = {'family': 'sans-serif', 'size': 30}
matplotlib.rc('font', **font)
fig = plt.figure(figsize=[42, 12])

# Save files with data and directory
if len(sys.argv) < 2: #if not console input
    #files = ["supsol-rf-mon-setA-nonorm.txt", "supsol-nn-monash-setA.txt", "supsol-cm-monash-setA.txt"]
    #files = ["supsol-rf-fruity-setA-nonorm.txt", "supsol-nn-fruity-setA.txt", "supsol-cm-fruity-setA.txt"]
    files = ["APUB-out-rf-fru.txt"]
    pathn = "viol-rf-forgit"
    set = "setA"
    monfru = "fru"

else:
    files = sys.argv[1:-3]
    print(files)
    pathn = sys.argv[-3]
    set = sys.argv[-2]
    monfru = sys.argv[-1]
    plots_lst = []


def boxplot_main(files, clf_name, pathn, set, monfru, FIGNAME, fig, boxcolor="cornflowerblue", alph=1.0):
    TITLE = 'Statistics of the residuals of features, {:}'.format(clf_name)
    figname = '{:}-{:}'.format(FIGNAME, clf_name)
    if monfru == "mon": TITLE = TITLE + str('Monash') + ',  ' + str(set)
    elif monfru == "fru": TITLE = TITLE + str('FRUITY') +',  ' + str(set)
    else: raise Exception("Give mon or fru")
    TITLE = 'Statistics of the residuals of features, comparison, FRUITY'
    figname = figname + '-' + str(monfru) + '-' + str(set)

    if not os.path.exists(pathn): os.makedirs(pathn)

    # Define all the directories
    dir_data = "/home/blans/Ba_star_classification/Ba_star_classification_data"
    fruity_mods = "models_fruity_dec"
    monash_mods = "models_monash"
    data_file = "all_data_w_err.dat"

    fruity_dir = os.path.join(dir_data, fruity_mods)
    monash_dir = os.path.join(dir_data, monash_mods)
    data_file = os.path.join(dir_data, data_file)

    # Load the stellar abundances
    dict_data = get_data_values(data_file)
    df_obs = conv_dict_to_df(dict_data)
    obs_elems = df_obs.columns.values.tolist()
    df_abunds = df_obs[[i for i in obs_elems if "err" not in i]]

    # Subtractions
    df_err = feature_subtract(df_obs[[i for i in obs_elems if "err" in i]], 1, plus=True)
    df_err = df_err[[i for i in df_err.columns if "Fe_err" not in i]]
    df_obs = feature_subtract(df_obs[[i for i in obs_elems if "err" not in i]], 1)
    df_obs = df_obs[[i for i in df_obs.columns if "/Fe" not in i]]
    ratio_names = df_obs.columns.values.tolist()

    # The fruity and monash models
    obs_elems = [i for i in obs_elems if "err" not in i] # observed elements
    fruity_models_dict = get_data_fruity(fruity_dir)
    df_fruity = conv_dict_to_df(fruity_models_dict).loc[:, obs_elems]
    monash_models_dict = get_data_monash(monash_dir)
    df_monash = conv_dict_to_df(monash_models_dict).loc[:, obs_elems]

        # Uncomment if necessary: calculating [Ce/Eu]
        # df_monash["Ce/Eu"] = df_monash["Ce/Fe"] - df_monash["Eu/Fe"]
        # delta = 1
        # df_monash["Ce/Eu_mod"] = np.log10(delta * (10**df_monash["Ce/Eu"] + (1-delta)))
        # df_monash.to_excel("monash_ce_eu.xlsx")
        # plt.hist(df_monash["Ce/Eu_mod"], bins=25)
        # #plt.hist(df_obs["Ce/Eu"], bins=25, alpha=0.5)
        # #plt.hist(df_abunds["Eu/Fe"], bins=25, alpha=0.5)
        # plt.hist(df_monash["Eu/Fe"], bins=25, alpha=0.5, label="Eu/Fe, models")
        #
        # plt.legend()
        # plt.show()

    cm = False
    if clf_name == "CM": cm = True
    predicted_models_dict = get_dict_predicted(files, cm)
    predicted_fru = predicted_models_dict['fruity']
    predicted_mon = predicted_models_dict['monash']

    # Load the red elements
    with open(os.path.join("../A_data_processing_and_plotting", "element_set.dat"), "r") as fread:
        for line in fread:
            lnlst = line.split()

            # Skip comments and empty lines
            if len(lnlst) == 0 or "#" in lnlst[0]:
                continue

            red_elements = lnlst
            break

    # Remove "/Fe"
    red_elements = [elem.split("/")[0] for elem in red_elements]
    red_elements.remove("Fe")
    peak1 = ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru']
        #peak1 = [val for val in peak1 if val in red_elements]
    peak2 = ['La', 'Ce', 'Nd', 'Sm', 'Eu']
        #peak2 = [val for val in peak2 if val in red_elements]

    stars_lst = df_obs.index.tolist()#[:10]

    resids = defaultdict(list)
    resids_abunds = defaultdict(list)
    stars_and_models = defaultdict(list)
    feh_lst = []
    mass_lst = []
    dil_lst = []
    stars_alllst = []

    # For each star, calcualte residuals
    print("Reading ratios in")
    for nstar in range(len(stars_lst)):
        print(nstar)
        starname = stars_lst[nstar]
        rowstar = df_obs.loc[starname]
        rowstar_abunds = df_abunds.loc[starname]
        rowstar_err = df_err.loc[starname]
        feh_obs = rowstar.pop('Fe/H')
        peak1p1_obs, peak1p2_obs, peak2p1_obs, peak2p2_obs, peak2p2_obs_r = peakfilter(rowstar, peak1, peak2)
            # peak1p1_err, peak1p2_err, peak2p2_err = peakfilter(rowstar_err, peak1, peak2)


        for file_now in files: # for each input file
            if "mon" in file_now:
                mode = "mon"
                possible_models = predicted_mon[starname]
            elif "fru" in file_now:
                mode = "fru"
                possible_models = predicted_fru[starname]

            for modelname, dil in [i[:2] for i in possible_models]:
                if mode == "mon":
                    model_now = df_monash.loc[modelname]
                    mass = float(modelname[modelname.find('_m') + 2:modelname.find('_m') + 6])
                elif mode == "fru":
                    model_now = df_fruity.loc[modelname]
                    mass = float(modelname[modelname.find('_m') + 2]) + float(modelname[modelname.find('_m') + 4]) / 10
                feh_model = model_now['Fe/H']
                stars_and_models['star'].append(starname)
                stars_and_models['model'].append(modelname)

                Eu_obs = df_abunds.loc[starname]['Eu/Fe']
                Eu_model = model_now.loc['Eu/Fe']
                Ce_Eu_obs = df_obs.loc[starname]["Ce/Eu"]

                model_dil_abunds = pd.DataFrame(apply_dilution_df(model_now, dil))
                model_dil = subtract_and_exclude(model_dil_abunds).iloc[0]
                peak1p1_mod, peak1p2_mod, peak2p1_mod, peak2p2_mod, peak2p2_mod_r = peakfilter(model_dil, peak1, peak2)

                model_dil_abunds = pd.Series(apply_dilution_df(model_now, dil)) # as series
                resid_now = rowstar - model_dil
                resid_abunds_now = rowstar_abunds - model_dil_abunds

                    # Uncomment if you want to filter for models with specific residuals or abundances
                    #if abs(feh_obs-feh_model) < 0.3 :
                    #if Eu_obs < 0.3 :
                for key in model_dil.keys():
                    resids[key].append(resid_now[key])
                for key in model_dil_abunds.keys():
                    resids_abunds[key].append(resid_abunds_now[key])
                feh_lst.append(feh_obs-feh_model)
                mass_lst.append(mass)
                dil_lst.append(dil)
                stars_alllst.append(starname)



    # DF for model residuals
    resids_abunds_df = pd.DataFrame.from_dict(resids_abunds)
    resids_abunds_df['Fe/H'] = feh_lst
    print("LEN")
    print(len(feh_lst))

    resids_df = pd.DataFrame.from_dict(resids)
    p1p1_abund, p1p2_abund, p2p1_abund, p2p2_abund, p2p2_abund_r = peakfilter(pd.Series(resids), peak1, peak2)
    df_all_resids = pd.concat([resids_abunds_df, pd.DataFrame(dict(p1p1_abund)), pd.DataFrame(dict(p1p2_abund)),
                            pd.DataFrame(dict(p2p2_abund))], sort=False, axis=1)
    if allthree:
        df_all_resids = pd.concat([resids_abunds_df, pd.DataFrame(dict(p1p1_abund)), pd.DataFrame(dict(p2p1_abund)),
                                   pd.DataFrame(dict(p2p2_abund))], sort=False, axis=1)
    df_all_resids.replace(to_replace='nan', value=np.nan, inplace=True)
    print(df_all_resids)


    # Save to a spreadsheet
    df_all_resids.to_excel('df_all_resids_filter.xlsx')
    df_median = df_all_resids.median()

        # Uncomment if necessary: Dropping out bad stars
        # df_cols = df_all_resids.columns.values.tolist()
        # bad_elems = ['Nb', 'Mo', 'Ru']
        # for elem in bad_elems:
        #     lst_features = [x for x in df_cols if ((x[:x.find('/')] == elem) or (x[x.find('/')+1:] == elem)) and ('Fe' not in x)]
        #     for nstar in range(len(df_all_resids['Fe/H'])):
        #         count = 0
        #         for feature in lst_features:
        #             f = df_all_resids.iloc[nstar][feature]
        #             med = df_median.loc[feature]
        #             if f < med: count += 1
        #         if count >= (len(lst_features)-1):
        #             print("feature")
        #             df_all_resids.drop(labels=[nstar], axis=0, inplace=True)
        #         print("")


    # PLOTTING
    # ---------------------------------------------------------------------------------------

    ax = plt.gca()
    if clf_name == 'CM':
        c = (0.86,0.07,0.24, 1)
        h = '.o'
        w = 0.8
    elif clf_name == 'NN':
        c = (0.3,0.45,0.87, 1)
        h = '\\'
        w = 0.6
    elif clf_name == 'RF':
        c = (0.19,0.8,0.19, 0.8)
        h = '/'
        w = 0.4
    else: print('Wrong clf name!')


    cols = df_all_resids.columns
    num_cols = len(cols.tolist())
        #df_all_resids = df_all_resids.apply(pd.to_numeric, errors='coerce')
        #df_all_resids = pd.DataFrame(df_all_resids, dtype='float')
        #df_all_resids = df_all_resids.astype(float)
        #df_all_resids.dropna(inplace=True)
        #df_all_resids[cols[1:]] = df_all_resids[cols[1:]].apply(pd.to_numeric, errors='coerce')

    color_median = matplotlib.colors.ColorConverter.to_rgb(boxcolor)
    color_median = tuple(i*0.7 for i in color_median)
    ax, bp_nice = df_all_resids.boxplot(grid=False, notch=True, patch_artist=True, showfliers=False, return_type="both",
            boxprops=dict(facecolor=boxcolor, color=color_median, alpha=alph),
            capprops=dict(color=boxcolor, linewidth=2),
            whiskerprops=dict(color=boxcolor, linewidth=2),
            medianprops=dict(color=color_median), widths=[w for i in range(num_cols)])


    red_elements.append('Fe')
    red_elements.append('H')
        #lst_reds = [x for x in df_cols if ((x[:x.find('/')] in red_elements) and (x[x.find('/')+1:] in red_elements))]
        #df_red_resids = df_all_resids[lst_reds]
    lst_nonreds = [x for x in cols if ((x[:x.find('/')] not in red_elements) or (x[x.find('/')+1:] not in red_elements))]
    df_red_resids = df_all_resids.copy()
    df_red_resids[lst_nonreds] = np.nan
    c = boxcolor

        # df_red_resids.boxplot(patch_artist=True, boxprops=dict(facecolor=c, alpha=alph), capprops=dict(color=c),
        #                       whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c), medianprops=dict(color=c),
        #                       notch=True, showfliers=False)

    toplim = ax.get_ylim()[1]
    toplim = 1.25
    botlim = ax.get_ylim()[0]
    botlim = -1.5
    #resids_df.boxplot()
    box_params = dict(boxstyle="round,pad=0.3", fc="lightcyan", ec="black", lw=2)
    vline_pos = len(resids_abunds)+0.5
    ax.text((1+vline_pos)/2, toplim, "[X/Fe]", ha="center", va="center", rotation=0, size=30, bbox=box_params)
    plt.axvline(vline_pos, color="black", lw=3)

    ax.text(vline_pos+len(p1p1_abund)/2, toplim, "[Peak 1 / Peak 1]", ha="center", va="center", rotation=0, size=30, bbox=box_params)
    vline_pos += len(p1p1_abund)
    plt.axvline(vline_pos, color="black", lw=2)

    ax.text(vline_pos+len(p1p2_abund)/2, toplim, "[Peak 2 / Peak 1]", ha="center", va="center", rotation=0, size=30, bbox=box_params)
    vline_pos += len(p1p2_abund)
    plt.axvline(vline_pos, color="black", lw=2)
    ax.text(vline_pos+len(p2p2_abund)/2, toplim, "[Peak 2 / Peak 2]", ha="center", va="center", rotation=0, size=30, bbox=box_params)

    plt.axhline(0, color='black', lw=1)

    plt.title(TITLE, size=42, weight='bold', y=1.06)
    plt.xticks(rotation=75)
    ax.set_ylim(botlim, toplim)
    plt.ylabel(r'[X/Y]$_{\mathbf{obs}}$ - [X/Y]$_{\mathbf{mod}}$', size=30, weight='bold')
    plt.tight_layout()
    resids_df.to_excel(r'./resids_ord.xlsx', index=False)

    df_all_resids['Fe/H'] = feh_lst
    df_all_resids['mass'] = mass_lst
    df_all_resids['star'] = stars_alllst
    df_all_resids['dil']  = dil_lst
    print('WRITE IT OUT')
    df_all_resids.to_csv('boxplot_resids.dat', index=False)
    df_all_resids.pop('mass')
    df_all_resids.pop('star')
    df_all_resids.pop('dil')
    #plt.close()


    # TABLE ------------
    df_all_resids.replace(to_replace='nan', value=np.nan, inplace=True)

    ax, bp = df_all_resids.boxplot(showmeans=True, return_type='both', showfliers=False, showcaps=True,
                                   showbox=True, boxprops=dict(alpha=0), whiskerprops=dict(color=(0,0,0,0), alpha=0),
                                   capprops=dict(alpha=0), flierprops=dict(alpha=0),
                                   medianprops=dict(alpha=0), meanprops=dict(alpha=0))

    categories = list(df_all_resids.columns) # to be updated
    medians = [round(item.get_ydata()[0], 2) for item in bp['medians']]
    means = [round(item.get_ydata()[0], 2) for item in bp['means']]
    minimums = [round(item.get_ydata()[0], 2) for item in bp['caps']][::2]
    maximums = [round(item.get_ydata()[0], 2) for item in bp['caps']][1::2]
    q1 = [round(min(item.get_ydata()), 2) for item in bp['boxes']]
    q3 = [round(max(item.get_ydata()), 2) for item in bp['boxes']]
    fliers = [item.get_ydata() for item in bp['fliers']]
    stats = [medians, means, q1, q3, minimums, maximums]
    zipped = list(zip(*stats))
    df_boxpl = pd.DataFrame(zipped, columns=['Median', 'Mean', 'Q1', 'Q3', 'Min.', 'Max.'])
    df_boxpl.index = categories
    #df_boxpl = df_boxpl.T
    df_boxpl.to_excel(r'./boxpl-data.xlsx')
    print(df_boxpl)

    tot_len = len(peak1p1_obs) + len(peak1p2_obs) + len(peak2p2_obs) + len(resids_abunds)



    def peakplot(series_toplot, pnpm_obs, boxtit, savename_end):
        '''A small plot containing only one peak / peak'''

        df_toplot = pd.DataFrame(dict(series_toplot))
        if "p2p2" in savename_end: fig = plt.figure(figsize=[70*(len(df_toplot.columns)/tot_len), 12])
        elif "abund" in savename_end: fig = plt.figure(figsize=[70*(len(df_toplot.columns)/tot_len), 12])
        else: fig = plt.figure(figsize=[70 * (len(df_toplot.columns) / tot_len), 12])

        ax = plt.gca()
        plt.axhline(0, color='black', lw=1)
        c='teal'
        df_toplot.columns = [r'$\Delta$['+x+']' for x in df_toplot.columns]
        sns.violinplot(data=df_toplot)
        sns.boxplot(data=df_toplot, boxprops={"zorder": 2, "alpha": 1}, width=0.2, color='black', showfliers=False, whis=0, notch=True)


        # Uncomment if necessary: used elements have red boxplots
        # lst_nonreds = [x for x in df_cols if
        #                ((x[:x.find('/')] not in red_elements) or (x[x.find('/') + 1:] not in red_elements))]
        # df_red_resids = df_all_resids.copy()
        # df_red_resids[lst_nonreds] = np.nan
        # c = "crimson"
        #
        # df_red_resids.boxplot(patch_artist=True,
        #                       boxprops=dict(facecolor=c, color=c),
        #                       capprops=dict(color=c),
        #                       whiskerprops=dict(color=c),
        #                       flierprops=dict(color=c, markeredgecolor=c),
        #                       medianprops=dict(color=c))

        ax.text((len(df_toplot.columns)-1) / 2, 1.3, boxtit, ha="center", va="center", rotation=0, size=30, bbox=box_params)
        ax.set_ylim(-1.5, 1.25)
        plt.title('{:} FRUITY'.format(clf_name), weight='bold', y=1.08)
        plt.xticks(rotation=75)
        plt.ylabel(r'[X/Y]$_{\mathbf{obs}}$ - [X/Y]$_{\mathbf{mod}}$', size=30, weight='bold')
        plt.tight_layout()
        savename = figname + savename_end #+ '-' + clf_name
        plt.savefig(os.path.join(pathn, savename), bbox_inches='tight')
        plt.close()

    peakplot(resids_abunds_df, peak1p1_obs, "[X/Fe]", "-abunds")
    peakplot(p1p1_abund, peak1p1_obs, "[Peak 1 / Peak 1]", "-p1p1")
    peakplot(p1p2_abund, peak1p2_obs, "[Peak 1 / Peak 2]", "-p1p2")
    peakplot(p2p2_abund, peak2p2_obs, "[Peak 2 / Peak 2]", "-p2p2")
    peakplot(p2p1_abund, peak2p1_obs, "[Peak 2 / Peak 1]", "-p2p1")

    return bp_nice


if allthree:
    bp_cm = boxplot_main([files[2]], 'CM', pathn, set, monfru, FIGNAME, fig, boxcolor="crimson")
    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=True,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=True) # labels along the bottom edge are off
    bp_nn = boxplot_main([files[1]], 'NN', pathn, set, monfru, FIGNAME, fig, alph=0.7)



bp_rf = boxplot_main([files[0]], 'RF', pathn, set, monfru, FIGNAME, fig, boxcolor="lime", alph=0.5)

ax = plt.gca()
plt.xticks(rotation=75)
ax.tick_params(axis="x", direction="in", length=20)
ax.grid(False)
labels = ['CM', 'NN', 'RF']
plt.tight_layout()

if allthree:
    ax.legend([bp_cm["boxes"][0], bp_nn["boxes"][0], bp_rf["boxes"][0]], ['CM', 'NN', 'RF'], loc='lower right')

plt.savefig(os.path.join(pathn, FIGNAME))