import sys, os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
import pandas as pd
import itertools
from processplot_data_lib import *
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
sys.path.insert(0, "/data_processing_and_plotting")

# PARAMETERS TO SET
distr_mode = 'dil' # 'dil', 'mass', 'gof'
mass_dil = False # if dil vs. mass 2D histograms
FIGNAME_DISTR = "dil-distr-supsol-fru-setA-DEC"
XLABEL = r"$\delta$ factor" #r"$\delta$ factor" #r"Mass [M$_{\odot}$]"
TITLE_DISTR = r"$\delta$ distribution, FRUITY, set A"
dir = "./try3/" # output directory
files = ["APUB_out-cm-fru.txt", "APUB_out-rf-fru.txt", "APUB_out-nn-fru.txt"]  # input files with classification results

distr_step = 0.05 # binsize -- mass: 0.5; dil: 0.1; gof: 5
distr_factor = float(1/distr_step)
distr_min = 0   # minimum value
distr_max = 1   # max. value -- ranges: mass: 1-5; dil: 0-1; gof: 1-100
intersect = 'on' # if to show the intersect of the three classifiers
distr_colors = ['crimson', 'lime', 'royalblue', 'crimson', 'lime', 'royalblue'] # colours of histograms
distr_hatch = ['.o', '\\', '/', '.o', '\\', '/'] # hatches of histograms

font = {'family': 'sans-serif', 'size': 30}
matplotlib.rc('font', **font)

def get_dict_predicted_massdistr(files):
    """ Make input dictionary to combine all files """

    dic_ = {}
    starnames = []
    # Go for file and line
    for file_, clf_name in files:
        dic_[clf_name] = (pd.DataFrame(columns=['star', 'mass', 'zed', 'mmix', 'label', 'dil', 'gof', 'proba', 'clf']))
        with open(file_, "r") as fread:
            cm = False # if closest method, there is no probability to read
            if "cm" in file_: cm = True
            for line in fread:
                lnlst = get_clean_lnlst(line, cm)

                # Skip lines without content
                if lnlst is None:
                    continue

                # Read the star name and add the sets in
                # the dictionaries if they were not there
                if "star" in lnlst:
                    star_name = lnlst[-1][:-1]
                    if star_name not in starnames:
                        # The starname set
                        starnames.append(star_name)
                else:
                    label = lnlst[0]
                    if "monash" in label:
                        mass = float(label[label.find("_m")+2:label.find("_m")+6])
                        mmix = float(label[label.find("_mix")+5:label.find("_mix")+13])
                        zed = float(label[label.find("_z")+2:])
                    elif "fruity" in label:
                        mass = float(label[label.find("_m") + 2:label.find("z")].replace("p", "."))
                        mmix = 0
                        if "zsun" not in label: zed = float(label[label.find("z") + 1:label.find("z") + 4].
                                                            replace("m", "E-"))
                        else: zed = 0.014
                    dic_[clf_name] = dic_[clf_name].append(dict(zip(dic_[clf_name].columns,
                                        [star_name, mass, zed, mmix, *lnlst, clf_name])), ignore_index=True)
    return dic_, starnames

def get_hist_points(intervals):
    """ Get histogram points by rounding to nearest halves """
    intervals_rounded = []
    all_possible_mass = []
    for ii in range(len(intervals)):
        if len(intervals[ii]) == 2:
            min_round = int(intervals[ii][0]*distr_factor)/distr_factor # round down
            max_round = int(intervals[ii][1]*distr_factor-distr_step)/distr_factor # round down
            intervals_rounded.append([min_round, max_round])
            min_now = min_round
            while max_round >= min_now:
                all_possible_mass.append(min_now)
                min_now += 0.5
        else: intervals_rounded.append([])
    return all_possible_mass


# if len(sys.argv) < 2:
#     s = "Incorrect number of arguments. "
#     s += f"Use: python3 {sys.argv[0]} <file1> [file2 ...]"
#     sys.exit(s)
# files = sys.argv[1:]

# Save files with data and directory
clf_names = []
for file in files:
    print()
    if "cm" in file:
        if "fru" in file: clf_names.append([file, "cm-fru"])
        elif "mon" in file: clf_names.append([file, "cm-mon"])
    elif "nn" in file:
        if "fru" in file: clf_names.append([file, "nn-fru"])
        elif "mon" in file: clf_names.append([file, "nn-mon"])
    elif "rf" in file:
        if "fru" in file: clf_names.append([file, "rf-fru"])
        elif "mon" in file: clf_names.append([file, "rf-mon"])

clf_full_names_dict = {"cm-fru": "CM FRUITY", "cm-mon": "CM Monash", "nn-fru": "NN FRUITY",
                       "nn-mon": "NN Monash", "rf-fru": "RF FRUITY", "rf-mon": "RF Monash"}
nfiles = len(files)

# Define all the directories
dir_data = "./Ba_star_classification_data"

pred_models, starnames = get_dict_predicted_massdistr(clf_names)
intervals = []

for star_now in starnames: # for each star, determine the inerval (bins in which they count)
    mass_currstar = defaultdict(list)
    for df_clf_name in pred_models:
        df_clf = pred_models[df_clf_name]
        df_currstar = df_clf[df_clf.star == star_now]
        mass_currstar[df_clf_name].append(df_currstar[distr_mode])

    mass_mins, mass_maxs = [], []
    for pair in itertools.combinations(mass_currstar.keys(), 2):
        mass_mins_pair, mass_maxs_pair = [], []
        for df_clf_name in pair:
            mass_mins_pair.append(min(mass_currstar[df_clf_name][0], default=np.nan))
            mass_maxs_pair.append(max(mass_currstar[df_clf_name][0], default=np.nan))

        if np.nan not in mass_mins_pair: mass_mins.append(max(mass_mins_pair, default=np.nan)) # min of pair interval
        else: mass_mins.append(np.nan)
        if np.nan not in mass_maxs_pair: mass_maxs.append(min(mass_maxs_pair, default=np.nan)) # max of pair interval
        else: mass_maxs.append(np.nan)

    absolute_min = np.nanmin(mass_mins)
    absolute_max = np.nanmax(mass_maxs)
    if absolute_max > absolute_min: interval_currstar = [absolute_min, absolute_max]
    else: interval_currstar = []
    intervals.append(interval_currstar)

all_possible_mass = get_hist_points(intervals)


# PLOTTNG -----------------------------------------------------------------
plt.figure(figsize=[12,9])
binbound = np.arange(distr_min-0.001, distr_max, distr_step)
nplot = 0
for df_clf_name in pred_models:
    mass_clf = []
    df_clf = pred_models[df_clf_name]
    for star_now in starnames:
        mass_currstar = []
        df_currstar = df_clf[df_clf.star == star_now]
        for mass_now in df_currstar[distr_mode]:
            mass_now = int(mass_now*distr_factor)/distr_factor # round down
            if mass_now not in mass_currstar: mass_currstar.append(mass_now)
        mass_clf.extend(mass_currstar)

    plt.hist(mass_clf, bins=binbound, label=clf_full_names_dict[df_clf_name], alpha=(1-nplot*0.25),
             color=distr_colors[nplot], hatch=distr_hatch[nplot])
    nplot += 1

if distr_mode != 'gof' and intersect == 'on':
    plt.hist(all_possible_mass, bins=binbound, edgecolor='yellow', lw=3, fill=False)
    plt.hist(all_possible_mass, bins=binbound, edgecolor='black', lw=3, linestyle=":", label="Intersection", fill=False)

ax=plt.gca()
plt.locator_params(axis='x', nbins=9)
labels = ax.get_yticklabels()
labels[0] = ""
ax.set_yticklabels(labels)

plt.title(TITLE_DISTR)
plt.legend(fontsize=26)
plt.xlabel(XLABEL)
plt.ylabel('Number of stars')
plt.xlim(distr_min, distr_max)
plt.tight_layout()
plt.savefig(FIGNAME_DISTR)
plt.show()
plt.close()

if mass_dil == True:
    df_clf_lst = []
    for df_clf_name in pred_models:
        df_clf_lst.append(df_clf_name)
        df_clf = pred_models[df_clf_name]
        df_clf['mass*dil'] = df_clf['mass']*df_clf['dil']
        df_massdil = df_clf[['mass', 'dil']].copy()
        mass_list = []
        [mass_list.append(x) for x in df_massdil['mass'] if x not in mass_list]
        mass_list.sort()

        plt.figure(figsize=[16, 10])
        plt.hist2d(df_massdil['mass'], df_massdil['dil'], bins=[np.arange(0.999, 5.1, 0.5),
                        np.arange(-0.001, 1, 0.1)], cmap=plt.get_cmap('cubehelix'))
        plt.colorbar()
        ax = plt.gca()

        box_params = dict(boxstyle="round,pad=0.3", fc="crimson", ec="black", lw=2)
        vline_pos = 4.5
        ax.text(4.4, 0.92, clf_full_names_dict[df_clf_name], ha="center", va="center", rotation=0, size=30,
                bbox=box_params)

        plt.xlabel('Tömeg')
        plt.ylabel(r'$\delta$')
        plt.title(r'Tömeg-$\delta$ hisztogram', weight="bold")
        plt.tight_layout()
        plt.savefig(f'hist2d-mass-dil-{df_clf_name}')


    fig = plt.figure(figsize=(24, 18))
    #fig = plt.figure(1, (6., 6.))
    plt.suptitle(r'Tömeg-$\mathbf{\delta}$ hisztogramok', y=0.9, weight="bold", size=42)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 2),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     cbar_mode="single",
                     cbar_location="right",
                     cbar_pad=0.1,
                     share_all=True
                     )
    for i in range(len(grid)):
        grid[i].set_aspect(2)

    hist_maxs = []
    for nplot in range(len(df_clf_lst)):
        hist = np.histogram2d(pred_models[df_clf_lst[nplot]]['mass'], pred_models[df_clf_lst[nplot]]['mass*dil'],
                          bins=[np.arange(0.999, 5.1, 0.5),
                            np.arange(-0.001, 5, 0.1)])
        hist_maxs.append(np.amax(hist[0]))
    hist_maxs.sort()
    hist_top = hist_maxs[-1]

    nplot = 0
    for ax in grid:
        h = ax.hist2d(pred_models[df_clf_lst[nplot]]['mass'], pred_models[df_clf_lst[nplot]]['dil'],
                      bins=[np.arange(0.999, 5.1, 0.5),
                        np.arange(-0.001, 0.9, 0.1)], cmap=plt.get_cmap('cubehelix'), vmin=0, vmax=hist_top)
        #im = plt.imshow(h)
        ax.text(4.35, 0.78, clf_full_names_dict[df_clf_lst[nplot]], ha="center", va="center", rotation=0, size=30,
                bbox=box_params)
        nplot += 1
    fig.colorbar(h[3], cax=grid.cbar_axes[0], orientation='vertical')
    fig.text(0.5, 0.04, 'Tömeg', ha='center', size=34)
    fig.text(0.002, 0.5, r'Keveredési faktor ($\delta$)', va='center', rotation='vertical', size=34)
    plt.tight_layout()
    plt.savefig('hist2d-subplots3')

for kk in range(len(intervals)):
    print(starnames[kk])
    print(intervals[kk])


