import numpy as np
import matplotlib.pyplot as plt
import struct, os, sys

from collections import Counter
from data_processing_and_plotting.process_data_lib import *

DIR = "data_processing_and_plotting"

def find_matches(D_nn, D_clos, mass_R=0.25, met_R=5.0):
    """
    Finding matches between the output files of different algorithms.
    """

    inputdicts = [D_nn, D_clos]
    # Initialise dictionary that holds all the matches:
    overlap = {}
    flagged_bad =  {}
    values = {}

    # Loop over the stars to find the matches
    for starname in D_clos.keys():

        nn_minmax_mass = []
        nn_minmax_met = []
        overlap[starname] = {}
        flagged_bad[starname] = {}

        # First get all the range from ANN in one set
        values[starname]= set()
        for filename in inputdicts:

            if filename == D_nn:

                for key_ in filename[starname].keys():
                    tup_ = filename[starname][key_]
                    for ii in range(len(tup_)):
                        values[starname].add(tup_[ii])

        # Then compare with overlap ranges of closest:
        for value in values[starname]:
            v_label = value[0].split("-")
            v_label2 = v_label[1].split("z")
            v_mass = v_label2[0][1:]
            v_met = '0.'+v_label2[1]
            nn_minmax_mass.append(v_mass)
            nn_minmax_met.append(v_met)

        nn_min_mass, nn_max_mass = min(nn_minmax_mass), max(nn_minmax_mass)
        nn_min_met, nn_max_met = min(nn_minmax_met), max(nn_minmax_met)

        # Try to find overlap regions by taking highest low and lowest high
        # Starting with mass:
        try:

            m_min_C = D_clos[starname]['range_ms'][0][0]
            m_max_C = D_clos[starname]['range_ms'][0][1]
            z_min_C = D_clos[starname]['range_zs'][0][0]
            z_max_C = D_clos[starname]['range_zs'][0][1]

            m_lowlim = max(float(nn_min_mass), float(m_min_C)) - mass_R
            m_toplim = min(float(nn_max_mass), float(m_max_C)) + mass_R

            # Is the highest low higher than the lowest high?
            # If yes, then check metalicity:
            if m_lowlim < m_toplim:
                z_lowlim = max(float(nn_min_met)/met_R, float(z_min_C)/met_R)
                z_toplim = min(float(nn_max_met)*met_R, float(z_max_C)*met_R)

                # Is the highest low higher than the lowest high?
                # if yes, then add to overlap dictionary
                if z_lowlim < z_toplim:
                    overlap[starname]['mass'] = (m_lowlim, m_toplim)
                    overlap[starname]['metallicity'] = (z_lowlim, z_toplim)
                    overlap[starname]['GoFs'] = D_clos[starname]['GoFs']

        # If there are no classifications in the closest algo,
        # then just flag star as bad
        # TODO what exception are we catching here?
        except:
            flagged_bad[starname]['nn'] = values[starname]
            flagged_bad[starname]['closest'] = D_clos[starname]['all_cla']

    # No overlap found? then add star to 'flagged-bad' dictionary
    for starname in D_clos.keys():
        if len(overlap[starname]) == 0 and len(flagged_bad[starname]) == 0:
            flagged_bad[starname]['nn'] = values[starname]
            flagged_bad[starname]['closest'] = D_clos[starname]['all_cla']

    # Remove empty keys from dictionaries
    for starname in D_clos.keys():
        if len(flagged_bad[starname]) == 0:
            flagged_bad.pop(starname)
        if len(overlap[starname]) == 0:
            overlap.pop(starname)

    return overlap, flagged_bad

def get_range_classifications(outpF, mod_mon, mod_fru):
    # Read every star
    starKey = None
    all_classes = {}

    # Gather all the info
    with open(outpF, "r") as fread:
        for line in fread:
            lnlst = line.split()

            # Get new key if any
            if "star" in line:
                starKey = lnlst[-1][:-1]
                all_classes[starKey] = []
            # Get parameters classifications
            if "Label" in line:
                # Get GoF and residual :
                label = lnlst[1]
                GoF = float(lnlst[6][:-1])
                all_classes[starKey].append((name_check(label, dir_=DIR), GoF))

    # Prepare and sort the data into small dictionaries
    range_ms = dict()
    range_zs = dict()
    gof = dict()
    for type_ in all_classes.keys():
        range_ms[type_] = []
        range_zs[type_] = []
        all_ms = []
        all_zs = []
        all_gofs = []
        gof[type_] = []
        for tuple_ in all_classes[type_]:
            split_label = tuple_[0].split("-")
            split_label2 = split_label[1].split("z")
            split_mass = split_label2[0][1:]
            split_met = '0.'+split_label2[1][:-1]
            # Print(split_mass, split_met)
            all_ms.append(split_mass)
            all_zs.append(split_met)
            all_gofs.append(tuple_[1])
        if len(all_ms)>0:
            range_ms[type_].append((min(all_ms), max(all_ms)))
            range_zs[type_].append((min(all_zs), max(all_zs)))
            gof[type_].append(min(all_gofs))

    # Fill final dictionary with all the info:
    D_range_classies = dict()
    for star in all_classes.keys():
        D_range_classies[star] = dict()
        D_range_classies[star]['all_cla'] = all_classes[star]
        D_range_classies[star]['range_ms'] = range_ms[star]
        D_range_classies[star]['range_zs'] = range_zs[star]
        D_range_classies[star]['range_zs'] = range_zs[star]
        D_range_classies[star]['GoFs'] = gof[star]

    return D_range_classies

def clean_for_table(match, round_n=4):
    """
    Turn duplicate into latex format
    """

    # This is for the mass and metallicity range:
    if ',' in match:
        match = match.replace(',', ' -')

    # Remove every char in "replace_chars"
    replace_chars = "()[]"
    for char in replace_chars:
        match = match.replace(char, '')

    # If we are dealing with a range
    if "-" in match:
        splitted = match.split("-")

        # Round to round_n digits maximum
        for ii in range(len(splitted)):
            if len(splitted[ii]) > round_n + 2:
                num = float(splitted[ii])
                splitted[ii] = f"{num:.{round_n}f} "

        match = "-".join(splitted)

    return match

def write_matches_into_latex_table(star_di, tab_name, tab_label, tab_caption,
                                   GoF=False):
    '''
    All results turned into a compilable latex tables
    '''

    # Creation of latex table
    # Step 1: Open file and write header:
    g = open(tab_name, 'w')

    # Step 2: Define caption and label
    tab_cap = 'This table lists the stars and their matched models.'
    tab_lab = 'tab:names'

    # Step 3: Write headings etc to table
    table_def = '\\begin{longtable}{l|llll}\n\\caption{'
    table_def += tab_cap + '}\\label{' + tab_lab + '}\\\ \n'
    g.write(table_def)

    tab_headings = 'Star & Mass range (Mo) & Metallicity range (Z) &'
    tab_headings += ' minimum GoF & agreement Cseh+2021\\\ \n'
    g.write(tab_headings)
    g.write('\\hline\n')

    # Step 4: Write the two sets of names
    for key, val in star_di.items():
        L_val = len(val)
        g.write(key)
        if L_val == 0:
            g.write(' & - & -\\\ \n')
        else:
            for key in val.keys():
                # Clean match needed?
                g.write(' & ' + clean_for_table(str(val[key])))
            g.write(' & \\\ \n')
        #g.write('\\hline\n')

    # Step 5: Write the final latex commands
    table_end = ('\\end{longtable}')
    g.write(table_end)
    g.close()

def main():
    """
    Load .txt files with output of classification algorithms, including
    goodness of fit. Compare the outputs per star:
    1) do the algorithms agree?
    2) is GoF above the threshold?
    3) make histogram with GoF values
    4) make .tex table with all outcomes?
    """

    # Get file names from input
    if len(sys.argv) < 3:
        s = "Incorrect number of arguments.\n"
        s += f"Use: python3 {sys.argv[0]} <output_1.txt> ... <output_n.txt>"
        sys.exit(s)

    files = sys.argv[1:]

    # Define all the directories
    dir_data = os.path.join(DIR, "Ba_star_classification_data")
    data_file = "all_abund_and_masses.dat"
    data_file = os.path.join(dir_data, data_file)
    star_vals = get_data_values(data_file)

    fruity_mods = "models_fruity"
    fruity_dir = os.path.join(dir_data, fruity_mods)
    models_F = get_data_fruity(fruity_dir)

    monash_mods = "models_monash"
    monash_dir = os.path.join(dir_data, monash_mods)
    models_M = get_data_monash(monash_dir)

    # Uncertainty ranges in matching:
    mass_R = 0.25
    met_R = 5.0

    D_range_classies = None
    D_stars = None
    for name in files:
        if 'closest' in name:
            D_range_classies = get_range_classifications(name, models_M,
                                                         models_F)
        elif 'nn' in name:
            D_files, D_stars = read_files_into_dicts(name)

    # Make sure we have enough
    if D_range_classies is None:
        s = "Stopping. Please, provide at least one closest classification."
        sys.exit(s)
    if D_stars is None:
        s = "Stopping. Please, provide at least one "
        s += "neural network classification."
        sys.exit(s)

    overlap, flagged_bad = find_matches(D_stars, D_range_classies, mass_R,
                                        met_R)

    print(len(flagged_bad))
    for star in flagged_bad.keys():
        print(star)
        for key in flagged_bad[star].keys():
            print(key)
            print(flagged_bad[star][key])
        print('---')

    # Set name, label, caption of table
    table_name = 'Latex_table_matchedstars.tex'
    table_label = 'tab:one'
    table_caption = 'caption check'

    # Write table with results
    write_matches_into_latex_table(overlap, table_name, table_label,
                                   table_caption, GoF=True)

if __name__ == "__main__":
    main()
