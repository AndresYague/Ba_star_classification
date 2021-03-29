import pandas as pd
from functools import reduce

def import_deC():
    # IMPORTING THE DATAFILES ---------------------------------------
    df_masscomp = pd.read_csv("inp_dec/Panda_masses_comp_agbm3.dat", delim_whitespace=True, index_col='name', na_values='-')
    df_dec_old = pd.read_csv("inp_dec/Panda_dec_all_measured_abund_mass.dat", delim_whitespace=True, index_col='name',
                             na_values='-')
    df_dec_err = pd.read_csv("inp_dec/Panda_errfile2.dat", delim_whitespace=True, index_col='name', na_values='---')
    df_feherr = pd.read_csv("inp_dec/Panda_feherr.dat", delim_whitespace=True, index_col='name', na_values='-',
                            usecols=['name', 'feh_err'])
    df_dec_new = pd.read_csv("inp_dec/Panda_data_28_newabund.dat", delim_whitespace=True, index_col='name', na_values='-',
                             usecols=['name', 'Eu', 'Rb', 'La', 'Sm', 'Sr', 'Nb', 'Mo', 'Ru', 'err_Ru', 'err_Eu', 'err_Mo', 'err_Nb', 'err_Sr', 'err_La', 'err_Sm', 'err_Rb'])
    # df_Rb = pd.read_csv("Panda_Rb_28.dat", delim_whitespace=True, index_col='name', na_values='-',
    #         usecols=['name', 'Rb'])

    # joining the tables ---------
    df_names = [df_masscomp, df_dec_old, df_dec_err, df_feherr, df_dec_new]  # name of dataframes
    df_all = reduce(lambda left, right: pd.merge(left, right, on=['name'], how='inner'), df_names)
    obs_elements = ['Na', 'Mg', 'Al', 'Si', 'Ca', 'Ti', 'Cr', 'Ni', 'Y', 'Zr', 'La', 'Ce', 'Nd', 'Rb', 'Sr', 'Sm',
                    'Eu', 'Nb', 'Mo', 'Ru' ]  # elements of old observations

    return df_all, obs_elements

def import_rest():
    # IMPORTING THE DATAFILES ---------------------------------------
    df_ref1 = pd.read_csv("inp_rest/aa_ref1.dat", na_values='-', delim_whitespace=True)
    df_ref4 = pd.read_csv("inp_rest/aa_ref4.dat", na_values='-', sep='\t')
    df_refs = pd.read_csv("inp_rest/aa_refs.dat", na_values='-', delim_whitespace=True)
    # OK CELLS TO BE CHECKED!!


    # joining the tables ---------
    df_ref1['name']=df_ref1['name'].astype(str)
    df_ref4['name']=df_ref4['name'].astype(str)
    df_refs['name']=df_refs['name'].astype(str)
    df_all = df_ref1.append(df_ref4, sort=False, ignore_index=True)
    df_all = df_all.append(df_refs, sort=False, ignore_index=True)
    df_all.set_index('name', inplace=True)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    obs_elements = ['Nb', 'Zr', 'C', 'N', 'O',	'Na', 'Mg',	'Rb', 'Sr',	'Y', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu',
                     'Li', 'Al', 'Si', 'S', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                     'Mo', 'Sn', 'Pd', 'Gd', 'Dy', 'Lu', 'Hf', 'Ta', 'Pb']
    return df_all, obs_elements
