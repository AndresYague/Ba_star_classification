from funcs_readin import *
from funcs_sorter import *
import numpy as np

def fruity_existing_files2(mass_dict, met_dict, model_type, patterndict, currmodels, starn):
    df_fr = []
    fr_rotvel = []  # for storing the rotation velocity values
    for mod_ind in range(len(currmodels['mod_files'])):

        # Separate the name of the file
        name_file = os.path.split(currmodels["mod_files"][mod_ind])[-1]

        currmodels['masses'][mod_ind] = patterndict[mass_dict][0][patterndict[mass_dict][1].index(
            name_file[0:4])]
        currmodels['mets'][mod_ind] = patterndict[met_dict][0][patterndict[met_dict][1].index(
            name_file[4:])]

        if name_file[name_file.find('_')+1:] == "000.dat":
            currmodels['mod_types'][mod_ind] = "fruity"
        elif name_file[name_file.find('_')+1:] == "ext.dat":
            currmodels['mod_types'][mod_ind] = "ext"
        elif name_file[name_file.find('_')+1:] == "T60.dat":
            currmodels['mod_types'][mod_ind] = "t60"
        else:
            currmodels['mod_types'][mod_ind] = "rot"

        fr_rotvel.append(name_file[-7:-4])
        currmodels['rotvel'][mod_ind] = name_file[-7:-4]

        df_fr.append(fruity_import(currmodels['masses'], currmodels[
            'mod_files']))  # Searching for matches in the measured and Fruity atomic numbers

    fr_mass = (currmodels['masses'])
    fr_fehs = (currmodels['mets'])

    # for zind in range(len(fr_fehs)):
    #     mind = 0
    #     fr_mass_formet, fr_fname, fr_mass_rej = interval_find(patterndict[mass_dict], curr_agbm, mass_tolerance, True, rej_file)
    #     if fr_mass_rej == True: rej_file.write(
    #         'Rejecting {:} models because all of them are farther than the tolerance in mass\n'.format(model_type))
    #
    #     while (mind < len(fr_mass_formet)):
    #         fr_fname_curr = "./fruity/{:}{:}".format(fr_fname[mind], fr_fname_end[zind])
    #         if os.path.exists(fr_fname_curr): fr_fname[mind] = fr_fname_curr  # name of FRUITY file to open
    #         else:
    #             del fr_fname[mind], fr_mass_formet[mind]
    #             mind -= 1
    #         mind += 1

    return df_fr, fr_mass, fr_fehs, fr_rotvel


def fr_labels2(fr_mass, fr_zed, mod_name, currmodels):
    fr_label_arr = []
    for zind in range(len(fr_mass)):
        #currkey = 'fr' + str(fr_mass[kk])
        #fr_massp = "M={:.2f}".format(fr_mass[kk])
        #fr_zed = fr_fname[kk][fr_fname[kk].index('z'):fr_fname[kk].index('z') + 4]
        fr_label = mod_name
        if currmodels['mod_types'][zind] == "ext":
            fr_label += r"M$=\,${:.2f} (ext), [Fe/H]$=\,${:5.2f}, ".format(fr_mass[zind], fr_zed[zind])
        elif currmodels['mod_types'][zind] == "rot":
            fr_label += r"M$=\,${:.2f} (r{:}), [Fe/H]$=\,${:5.2f}, ".format(fr_mass[zind], currmodels['rotvel'][zind][1:], fr_zed[zind])
        elif currmodels['mod_types'][zind] == "t60":
            fr_label += r"M$=\,${:.2f} (T60), [Fe/H]$=\,${:5.2f}, ".format(fr_mass[zind], fr_zed[zind])
        else:
            fr_label += r"M$=\,${:.2f} ( -- ), [Fe/H]$=\,${:5.2f}, ".format(fr_mass[zind], fr_zed[zind])
        #fr_label += "{:}".format(r"M$_{\mathrm{mix}}$ = - , ")
        fr_label += r" $\delta=\,$" + "{:.2f}".format(currmodels['dils'][zind])
        fr_label_arr.append(fr_label)
    return fr_label_arr




