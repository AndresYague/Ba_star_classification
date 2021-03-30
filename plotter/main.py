import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, FixedLocator, AutoMinorLocator)
import funcs_datastruct, funcs_sorter, funcs_fruity_models, funcs_chis, funcs_readin
import os, sys
import pandas as pd
import numpy as np

def get_clean_lnlst(line):
    """
    Clean the input data so it's easier to handle
    """

    # Split line
    lnlst = line.split()

    # Return proper value
    if "Label" in line:
        return [lnlst[1], lnlst[7]]
    elif "star" in line:
        return lnlst
    else:
        return None

def get_dict_models(files):
    """
    Make input dictionary to combine all files
    """

    # Initialize dictionary
    dict_ = {}
    dict_["fruity"] = {}
    dict_["monash"] = {}

    # Keep an inventory of repeated models
    repeated = {}
    repeated["fruity"] = {}
    repeated["monash"] = {}

    # Go for file and line
    for file_ in files:
        with open(file_, "r") as fread:

            for line in fread:
                lnlst = get_clean_lnlst(line)

                # Skipe lines without content
                if lnlst is None:
                    continue

                # Read the star name and add the sets in
                # the dictionaries if they were not there
                if "star" in lnlst:
                    star_name = lnlst[-1][:-1]
                    if star_name not in dict_["fruity"]:

                        # The starname set
                        dict_["fruity"][star_name] = set()
                        dict_["monash"][star_name] = set()

                        # Add the list to the repeated models
                        repeated["fruity"][star_name] = []
                        repeated["monash"][star_name] = []

                # Add this line in fruity or monash
                else:
                    if "fruity" in lnlst[0]:
                        type_ = "fruity"
                    elif "monash" in lnlst[0]:
                        type_ = "monash"

                    # Check if repeated to skip
                    if lnlst[0] in repeated[type_][star_name]:
                        continue

                    # Add this model here to avoid repeating it
                    repeated[type_][star_name].append(lnlst[0])

                    # Add to the set
                    dict_[type_][star_name].add(tuple(lnlst))

    return dict_

def model_names_read(dict_models, type_):
    """
    Create the pandas table
    """

    # Lists
    lst_tmp1 = []
    dils = []
    starnames = []
    mods = []
    masses = []
    mets = []
    mod_types = []
    rotvel = []
    mmix = []
    num_star = -1

    # Select dictionary
    sub_dict = dict_models[type_]

    for star in sub_dict:
        starnames.append(star)
        num_star += 1
        [x.append(y) for x, y in zip([mods, dils, masses, mets, mod_types,
            rotvel, mmix], [[] for i in range(7)])]
        lst_tmp1.append({'starname': starnames[num_star], 'mod_files': mods[num_star],
                         'masses': masses[num_star], 'mets': mets[num_star],
                         'dils': dils[num_star],
                         'mod_types': mod_types[num_star],
                         'rotvel': rotvel[num_star], 'mmix': mmix[num_star]})

        for lnlst in sub_dict[star]:

            # Separate lines in fruity or monash
            if type_ == "fruity":
                surfname = lnlst[0][lnlst[0].find('_')+1:] + ".dat"
                file_name = os.path.join(type_, surfname)

                mods[num_star].append(file_name)
                [x.append(y) for x, y in zip([masses[num_star], mets[num_star],
                                              rotvel[num_star], mmix[num_star],
                                              mod_types[num_star]], [0, 0, 0, 0, ""])] # will be found later with fruity_existing_files2 func

            elif type_ == "monash":
                surfname = "surf_" + lnlst[0][lnlst[0].rfind('_')+1:] + ".dat"
                file_name = os.path.join(type_, surfname)

                mods[num_star].append(file_name)
                mets[num_star].append(patterndict['mon_met'][0][patterndict['mon_met'][1].index(surfname)])

                pos = lnlst[0].find("mix_")+4 # position of m_mix data in input
                mmix[num_star].append(float(lnlst[0][pos:lnlst[0].find("_", pos)]))
                pos = lnlst[0].find("_m")+2
                masses[num_star].append(float(lnlst[0][pos:lnlst[0].find("_", pos)]))
                [x.append(y) for x, y in zip([rotvel[num_star],
                                              mod_types[num_star]], [0, ""])] # those are for FRUITY

            dils[num_star].append(float(lnlst[1]))

        df_models = pd.DataFrame(lst_tmp1)
        df_models.set_index('starname', inplace=True)

    return df_models

if len(sys.argv) < 3:
    s = "Incorrect number of arguments. "
    s += f"Use: python3 {sys.argv[0]} <file1> [file2 ...] <directory>"
    raise Exception(s)

# Save files with data and directory
files = sys.argv[1:-1]
pathn = sys.argv[-1]

# Sort files in fruity and monash
dict_models = get_dict_models(files)

# Load the red elements
with open(os.path.join("..", "element_set.dat"), "r") as fread:
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

# general settings
rest = False # False if deC, True if Rest
eps = False      # True if format is eps, False if png
nolegend = False  # if True, no legend is plotted
rej_file = open("rej_models_moremass_extra_models_dec_newdata.txt", "w")  # rejection file
res_file = open("res_rest_Nb_newdata.dat", "w")
bestres = open("bestres_rest_Nb_newdata.dat", "w")
mass_tolerance = 0.3  # tolerance in mass -- plus AND minus
feh_tolerance = 0.5  # tolerance in fe/h
dil_tolerance = 0.9  # tolerance in dilution
chi_start = 37 # first peak: 37 (Rb), only second peak: 57 (La)

# plotting settings
figsiz = [60, 16]  # for paper: [28,10]
errbcolor = 'yellow'  # color of deC measured errorbar plot
erredgecolor = 'midnightblue'  # color of deC measured errorbar plot border
pmin = 2  # gap between atomic numbers on plot
xaxlim, yaxlim, y2axlim = [4.2, 83.6], [-1.3, 2.5], [-0.75, 0.75]  # axis limits
font = {'family': 'sans-serif', 'size': 52}  # for paper: 42
plt.rc('font', **font)

# atomic numbers for x axis of the plot
df_atnum = pd.read_csv("inp_model_patterns/atomic_nums.dat", delim_whitespace=True, index_col='Z')

# ----------------------------------------------------------------
df_all, obs_elements = funcs_datastruct.import_deC()

# IMPORTING THE MODEL PATTERNS --------------------------------------
patternarr_name = ['fr_mass', 'fr_met', 'mon_met', 'fr_mass_ext', 'fr_met_ext', 'fr_mass_rot', 'fr_met_rot',
                   'fr_mass_T60', 'fr_met_T60']  # name of each pattern type
patterndict = dict()  # dictionary to hold the pattern-value for each pattern type
dir_pattrn = "inp_model_patterns"
patternfiles = ["fruity_beginnings.dat", "fruity_endings.dat",
                "monash_beginnings.dat", "fruity_beginnings_ext.dat",
                "fruity_endings_ext.dat", "fruity_beginnings_rot.dat",
                "fruity_endings_rot.dat", "fruity_beginnings_T60.dat",
                "fruity_endings_T60.dat"]  # files consisting of pattern-value
patternfiles = [os.path.join(dir_pattrn, x) for x in patternfiles]

for nn in range(len(patternarr_name)):  # for each pattern type
    patterndict[patternarr_name[nn]] = funcs_sorter.namefinder_import(patternfiles[nn])

mod_fr = model_names_read(dict_models, "fruity")
mod_mon = model_names_read(dict_models, "monash")

starnames_lst = df_all.index.values  # all starnames into a list
for starn in starnames_lst:  # for each star
    print(starn)
    #curr_agbm = df_all.loc[starn, 'agb_ini']  # current agb_ini mass
    curr_agbm = df_all.loc[starn, 'mass_try']  # current agb_ini mass
    curr_feh = df_all.loc[starn, 'Fe/H']  # current fe/h
    curr_feh_err = df_all.loc[starn, 'feh_err']
    curr_ce_dec = 10 ** df_all.loc[starn, 'Ce']  # current deC Ce value

    # FRUITY abundances ------------------------------------------------
    feh_nanerr = False
    if np.isnan(curr_feh_err):
        curr_feh_err = 0.15
        feh_nanerr = True
    vars = [patterndict, curr_agbm, mass_tolerance, rej_file, curr_feh, curr_feh_err, feh_tolerance]

    currmodels_fr = mod_fr.loc[starn]
    df_fr, fr_mass, fr_feh, fr_rotvel = funcs_fruity_models.fruity_existing_files2('fr_mass', 'fr_met', 'FRUITY', patterndict, currmodels_fr, starn)

    currmodels_mon = mod_mon.loc[starn]
    mon_metnum = 0
    mon_masses_inzfile, mon_mass_matching, mon_mmixes = ([] for k in range(3))
    df_moni, df_moni_loge, df_monf, df_monf_loge = ([] for k in range(4))

    for feh in range(len(currmodels_mon['mets'])):
        mon_masses_inzfile, mon_inabund_pos, mon_finabund_pos = funcs_readin.monash_masses_inzfile(currmodels_mon['mod_files'][feh])
        mon_mass_filepos, mon_mmix = funcs_readin.monash_matching_finalpos(mon_masses_inzfile, currmodels_mon['masses'][feh],
                                             currmodels_mon['mmix'][feh],   mon_inabund_pos, mon_finabund_pos)
        mon_mmixes.append(mon_mmix)

        #if (currmodels_mon['mets'][feh] != currmodels_mon['mets'][feh-1]) and feh != 0: mon_metnum = 0

        df_moni.append(funcs_readin.monash_elements(mon_metnum, 'e(X)', currmodels_mon['mod_files'][feh], mon_mass_filepos[0], currmodels_mon['masses'],
                                        currmodels_mon['mmix'], rej_file))  # final [X/Fe]
        df_monf.append(funcs_readin.monash_elements(mon_metnum, 'e(X)', currmodels_mon['mod_files'][feh], mon_mass_filepos[1], currmodels_mon['masses'],
                                        currmodels_mon['mmix'], rej_file))  # final [X/Fe]

        mon_metnum += 1



    # PRE-PLOTTING --------------------------------------------------------
    # FRUITY labels
    fr_label_arr = funcs_fruity_models.fr_labels2(fr_mass, fr_feh, "{:10}".format("FRUITY:"), currmodels_fr)

    # labels for Monash (mon_label_arr, list)
    mon_label_arr = []  # array for the labels

    for kk in range(len(currmodels_mon['masses'])):  # if no mmixes left after dilution rejection, does nothing
        if currmodels_mon['mmix'][kk] < 0.001: mon_mlab = r"M$=\,${:.2f} ({:>2.0f})$\,$".format(currmodels_mon['masses'][kk], currmodels_mon['mmix'][kk]*10000)
        else: mon_mlab = r"M$=\,${:.2f} ({:>2.0f})".format(currmodels_mon['masses'][kk], currmodels_mon['mmix'][kk]*10000)
        mon_zlab = r"[Fe/H]$=\,${:5.2f}".format(currmodels_mon['mets'][kk])
        mon_deltalab = r" $\delta=\,$" + "{:.2f}".format(currmodels_mon['dils'][kk])
        mon_label = "{:7} {:5}, {:4}, {:}".format("Monash: ", mon_mlab, mon_zlab, mon_deltalab)
        mon_label_arr.append(mon_label)

    # label for deC measurements
    dec_massjor = r"M$_{{\mathrm{{Ba}}}}=\,${:.2f}$^{{{:.1}{:.1f}}}_{{{:.1f}}}$".format(df_all.loc[starn, 'Ba_mass'], '+',
                                                                                        df_all.loc[starn, 'err+'],
                                                                                        df_all.loc[starn, 'err-'])
    dec_agb_inimass = r"M$_{{\mathrm{{AGB, ini}}}}=\,${:.2f}".format(df_all.loc[starn, 'agb_ini'])
    if feh_nanerr == True: curr_feh_err = np.nan
    dec_fehdec = r"[Fe/H]$=\,${:.2f} $\pm$ {:.2f}".format(curr_feh, curr_feh_err)
    dec_label = "{:5}, {:5}, {:5}".format(dec_massjor, dec_agb_inimass, dec_fehdec)


    # deC measurements dataframe - df_dec_toplot
    df_dec_toplot = pd.DataFrame(
        df_all.loc[starn, obs_elements])  # reading in the deC measurements for current star to a df
    df_dec_toplot.index.names = ['element']  # the index is renamed to 'elements'
    df_dec_toplot.rename(columns={starn: 'deC_abund'}, inplace=True)  # the column is now 'deC_abund'
    df_dec_toplot = pd.merge(df_dec_toplot, df_atnum, right_on='element', how='inner',
                             left_index=True)  # associating with atnums
    obs_elements = df_dec_toplot['element'].tolist()  # deC elements now in the order of the df

    # filling up the df_dec_toplot with the errors
    keylist_df = df_all.keys().tolist()
    errors_lst = []  # key number of error values
    for kk in range(len(obs_elements)):
        if obs_elements[kk] != 'Fe/H':  # they do not have errors
            errors_lst.append(df_all.loc[starn, 'err_' + str(obs_elements[kk])])
        else:
            errors_lst.append(np.nan)
    df_dec_toplot['error'] = errors_lst  # filling up the df_dec_toplot df with the errors


    # PLOTTING =============================================================================================
    fig = plt.figure(figsize=figsiz)
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[0:2, :])
    ax2 = plt.subplot(gs[2, :], sharex=ax1)

    # deC points
    df_dec_toplot['deC_abund'] = df_dec_toplot['deC_abund'].astype(float)  # convert everything to float
    df_dec_toplot['error'] = df_dec_toplot['error'].astype(float)
    ax1.errorbar(df_dec_toplot.index, df_dec_toplot['deC_abund'], yerr=df_dec_toplot['error'], color=errbcolor,
                 mec=erredgecolor, ecolor=erredgecolor, label=dec_label, marker='*',
                 linestyle='', markersize=35, elinewidth=4, markeredgewidth=2, capsize=4, zorder=5)

    # red starred ones
    print(df_dec_toplot)
    atnums = list(df_dec_toplot.index.values)
    found = 0
    for ind in range(len(df_dec_toplot['element'])):
        elem = df_dec_toplot.iloc[ind]['element']
        if elem in red_elements:
            found += 1
            yerr_val = df_dec_toplot.iloc[ind]['error']
            if not yerr_val > 0: yerr_val = 0.5
            ax1.errorbar(atnums[ind], df_dec_toplot.iloc[ind]['deC_abund'], yerr=yerr_val, color='red',
                         mec=erredgecolor, ecolor='red', marker='*',
                         linestyle='', markersize=35, elinewidth=4, markeredgewidth=2, capsize=4, zorder=5)

    if found != len(red_elements): print("EZZAAAZZZZZZ")

    # FRUITY
    nplotted_all = 0

    # dilute the models
    chiok_arr = [0 for i in range(len(df_dec_toplot[(df_dec_toplot.index >= chi_start) & (df_dec_toplot.index <= 63)]['deC_abund']
                           [df_dec_toplot['deC_abund'].notna()]))]
    chiok_arr, fr_plotarr, df_fr_res_arr, modelnum_fr = funcs_chis.chi_fruity2(len(fr_mass), fr_mass, currmodels_fr['dils'],
                                 chiok_arr, chi_start, df_fr, df_dec_toplot, fr_label_arr)

    # plot them
    coloridx_fr = plt.cm.cool(np.linspace(0.1, 1, num=len(fr_mass) + 1))  # colormap distribution for Fruity
    for zind in range(len(fr_mass)):
        ax1.plot(df_fr[zind].index, fr_plotarr[zind], color=coloridx_fr[zind], linestyle='-', linewidth=3,
                 label=fr_label_arr[zind],  # + " - {:^3.0f}".format(chis_rank[nplotted_all]+1)		# chi2 kiiras
                 zorder=3, marker='v', alpha=0.8, markersize=15)
        ax1.plot(df_fr[zind].index, fr_plotarr[zind].interpolate(), color=coloridx_fr[zind], linestyle='-',
                 linewidth=3, zorder=3, marker='', alpha=0.8, markersize=15)  # interpolates between missing data
        ax2.plot(df_fr_res_arr[zind].index, df_fr_res_arr[zind]['res_dil'], color=coloridx_fr[zind],
                 marker='v',
                 linestyle='', markersize=20, markeredgewidth=1, zorder=3, alpha=0.8)

    # MONASH
    chiok_arr, mon_plotarr, df_mon_res_arr, modelnum_mon = funcs_chis.chi_res_mon2(currmodels_mon['masses'], currmodels_mon['mmix'],
                                currmodels_mon['dils'], chiok_arr, chi_start, df_moni, df_monf, df_dec_toplot, mon_label_arr)

    coloridx_mon = np.linspace(0, 1, num=modelnum_mon)  # colormap distribution for Monash
    for mod in range(modelnum_mon):
        ax1.plot(df_monf[0].index, mon_plotarr[mod], color=plt.cm.jet(coloridx_mon[mod]),
                 label=mon_label_arr[mod], #+ " - {:^3.0f}".format(chis_rank[nplotted_all]+1),	#chi2 kiiras
                 zorder=3, linestyle='--', marker='o', markersize=15, alpha=0.8, linewidth=3)
        ax2.plot(df_mon_res_arr[mod].index, df_mon_res_arr[mod]['res_dil'], color=plt.cm.jet(coloridx_mon[mod]), zorder=5,
                 marker='o', linestyle='',
                 markersize=20, markeredgewidth=1, alpha=0.8)

    # df_ok.dropna(inplace=True, how='all')

    # Setting the ticks of atomic names and vertical lines --------------------------------------
    toptick_labels, bottick_labels = [], []
    atnum_counter = int(xaxlim[0]) + 2
    while atnum_counter < max(df_atnum.index):
        toptick_labels.append(df_atnum.loc[atnum_counter, 'element'])
        bottick_labels.append(df_atnum.loc[atnum_counter - pmin, 'element'])

        ax1.axvline(x=df_atnum.index[atnum_counter - 1], color='lightgrey', zorder=1)
        ax1.axvline(x=df_atnum.index[atnum_counter - 1 + pmin], color='lightgrey', zorder=1, linestyle='dashed')
        ax2.axvline(x=df_atnum.index[atnum_counter - 1], color='lightgrey', zorder=1)
        ax2.axvline(x=df_atnum.index[atnum_counter - 1 + pmin], color='lightgrey', zorder=1, linestyle='dashed')

        atnum_counter += pmin * 2  # next toptick_label

    # ticks settings
    maj_ticks = np.linspace(xaxlim[0], xaxlim[1], 2*pmin)
    ax2.xaxis.set_major_locator(MultipleLocator(2 * pmin))
    #ax2.xaxis.set_major_locator(FixedLocator(maj_ticks))
    ax2.set_xticklabels(bottick_labels)
    ax2.xaxis.set_minor_locator(MultipleLocator(pmin))
    ax2.set_xticklabels(toptick_labels, minor=True)
    ax2.xaxis.set_tick_params(which='major', pad=42)

    # horizontal lines
    ax1.axhline(y=0, color='silver', linestyle='--', zorder=0)
    ax2.axhline(y=0, color='k', linestyle='-')
    ax2.axhline(y=0.2, color='k', linestyle='dashed')
    ax2.axhline(y=0.4, color='k', linestyle=':')
    ax2.axhline(y=-0.2, color='k', linestyle='dashed')
    ax2.axhline(y=-0.4, color='k', linestyle=':')

    # Setting axis limits, saving the figure ---------------------------------------------------------------------------
    ax2.set_ylabel('residual')
    ax1.set_ylabel('[X/Fe]')
    if rest == True:
        curref = df_all.loc[starn, 'ref']
        plt.title("{:} ({:.0f})".format(starn, curref), size=60, weight='bold', y=3.0)
    else:
#        plt.title("{:} ({:})".format(starn, "deC"), size=45, weight='bold', y=3.0)
        plt.title("{:}".format(starn), size=60, weight='bold', y=3.0)
    fig.subplots_adjust(hspace=0)

    plt.setp(ax1, xlim=xaxlim)  # ylim=[ymin - extray, ymax + extray + legend_height]
    plt.setp(ax2, xlim=xaxlim, ylim=y2axlim)
    fig.align_ylabels([ax1, ax2])

    ax1.tick_params(direction='out', top=False, right=True, which='major', width=1.5, length=6)
    ax1.tick_params(direction='out', top=False, right=True, which='minor', width=1.5)
    ax2.tick_params(direction='out', top=False, right=True, which='major', width=1.5, length=6)
    ax2.tick_params(direction='out', top=False, right=True, which='minor', width=1.5)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    # put the label of deC to top
    handles, labels = ax1.get_legend_handles_labels()
    handles.insert(0, handles[nplotted_all])
    del handles[nplotted_all+1]
    labels.insert(0, labels[nplotted_all])
    del labels[nplotted_all+1]
    ax1.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, -0.75), ncol=2)

    if nolegend == False:
        if eps == False:
            figname = starn + '.png'
        elif eps == True:
            figname = starn + '.eps'
    elif nolegend == True:
        ax1.get_legend().remove()
        if eps == False:
            figname = starn + '_noleg.pdf'
        elif eps == True:
            figname = starn + '_noleg.eps'

    # saving figure
    if not os.path.exists(pathn): os.mkdir(pathn)
    #plt.tight_layout()
    os.chdir(pathn)  # change directory to given path
    if eps == True: plt.savefig(figname, format='eps', transparent=False, dpi=fig.dpi)
    elif eps == False: plt.savefig(figname, transparent=False, bbox_inches='tight', dpi=fig.dpi)
    os.chdir('..')  # step out from directory
    plt.close()
    plt.clf()
    rej_file.write('\n')
