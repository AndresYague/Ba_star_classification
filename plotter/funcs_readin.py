import pandas as pd
import os.path
from funcs_sorter import *
#import masses_plots_pandas_moremass_all2_nov



def fruity_import(optimal_masses, fname_arr):
    # df_fr: dataframe: index - element name, key: fr_atnum (atomic number) and 0,1... (abundances for each found mass)
    df_out = pd.DataFrame()
    for mid in range(len(optimal_masses)):
        cols = pd.read_csv(fname_arr[mid], delim_whitespace=True, nrows=1).columns
        if mid == 0: # initialising the dataframe
            df_out = pd.read_csv(fname_arr[mid], delim_whitespace=True, index_col='Z', na_values='\\N',
                                 usecols=['[El/Fe]', 'Z', cols[-1]])
            df_out.rename(columns={'[El/Fe]': 'element', cols[-1]:'fr'+str(optimal_masses[mid])}, inplace=True)
        else:
            df_new = pd.read_csv(fname_arr[mid], delim_whitespace=True, index_col='Z', na_values='\\N',
                                 usecols=['Z', cols[-1]])
            df_new.rename(columns={cols[-1]: 'fr'+str(optimal_masses[mid])}, inplace=True)
            df_out = pd.merge(df_out, df_new, on=['Z'], how='inner')
    return df_out


def monash_masses_inzfile(fname):
        # Finding the masses stored in the file, then choose the closest ones ---------
    monashinp = open(fname, "r")
    masses_inzfile = [[], []]    # masses and M_mixes included in the Monash (closest z) datafile (list in list for the format of interval_find)
    inabund_pos_all, finabund_pos_all = [], []   # position of initial and final abundances in file
    linecount = 0  # counter for line in file
    novrej = False   # boolean - True if N_ov or M_mix is 0 for model (then the model is not used)
    for line in monashinp:
        linecount += 1
        lnlst = line.split()
        if '# Initial mass = ' in line:
            #if 'M_mix = 0.00E+00' not in line and 'N_ov = 0.0' not in line:
            masses_inzfile[0].append(float(lnlst[4][:-2])) # initial mass
            if len(lnlst) == 14:
                masses_inzfile[1].append(float(lnlst[13])) # M_mix
            else: # if not last element in lnlst (because there is N_ov), it includes a ',' at the end
                masses_inzfile[1].append(float(lnlst[13][:-1]))
            #else: novrej = True
        if '# Initial abundances' in line:  # it will be after every 'Initial mass' line
            if novrej == False:
                inabund_pos_all.append(linecount)
        if '# Final abundances' in line: # it will be after every 'Initial mass' line
            if novrej == False:
                finabund_pos_all.append(linecount)
            else: novrej = False # if novrej was True, set it back to default False
    return masses_inzfile, inabund_pos_all, finabund_pos_all


def monash_matching_finalpos(masses_inzfile, masses_matching, mmix_matching, inabund_pos_all, finabund_pos_all):
    mon_mass_filepos = []  # position of tolerated masses in file
    mmixes = []
    count = 0
    for kk in range(len(masses_inzfile[0])):
        if masses_inzfile[0][kk] == masses_matching and masses_inzfile[1][kk] == mmix_matching:
            mon_mass_filepos.append(inabund_pos_all[kk])  # position of 'Final abundances' of current mass
            mon_mass_filepos.append(finabund_pos_all[kk])  # position of 'Final abundances' of current mass
            mmixes.append(masses_inzfile[1][kk])
            count += 1
        #if count == len(masses_matching):  # no more masses found
         #     break
    return mon_mass_filepos, mmixes


def monash_elements(colfound, header, fname, matching_mass_filepos, mass_list, mmix_list, rej_file):
    # Read the file again, search for the last DU of that mass and import abundances ---------------
    newcolname = str(mass_list[colfound]) + 'mix' + str(mmix_list[colfound]) # name of current column
    df_out = pd.read_csv(fname, delim_whitespace=True, na_values='-', skiprows=matching_mass_filepos,
                         nrows=81, usecols=['#', 'El', header], index_col='El')  # index: atnum
    df_out.rename(columns={'#': 'element', header: newcolname}, inplace=True)
    df_out.index.names = ['Z']

    return df_out


def monash_colname(masses_matching, index):
    return '{:},{:}'.format(masses_matching[0][index], masses_matching[1][index])
