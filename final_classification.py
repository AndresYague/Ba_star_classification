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
        if len(values[starname])<1:
            nn_minmax_mass.append(0)
            nn_minmax_met.append(-10)
        else:
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
            if m_lowlim <= m_toplim:
                z_lowlim = max(float(nn_min_met)/met_R, float(z_min_C)/met_R)
                z_toplim = min(float(nn_max_met)*met_R, float(z_max_C)*met_R)

                # Is the highest low higher than the lowest high?
                # if yes, then add to overlap dictionary
                if z_lowlim <= z_toplim:

                    overlap[starname]['mass'] = (m_lowlim, m_toplim)
                    overlap[starname]['metallicity'] = (z_lowlim, z_toplim)
                    overlap[starname]['GoFs'] = D_clos[starname]['GoFs']

        # If there are no classifications in the closest algo,
        # then just flag star as bad
        except IndexError:
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
        '''if len(overlap[starname]) == 0:
            overlap.pop(starname)'''

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
        splitted = match.split(" - ")

        # Round to round_n digits maximum
        for ii in range(len(splitted)):
            if len(splitted[ii]) > round_n + 2:
                num = float(splitted[ii])
                splitted[ii] = f"{num:.{round_n}f}"

                # Remove the trailing zeroes
                jj = 1
                while splitted[ii][-1] == "0":
                    splitted[ii] = f"{num:.{round_n - jj}f}"
                    jj += 1

        match = " - ".join(splitted)

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
            g.write(' & - & - & - & -\\\ \n')
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

def convert_met(zzraw):
    """
    Convert metallicity to feH, zz is a string
    """

    # Define several quantities to use
    zz_sun = 0.014
    hyd = 0.75
    hyd_sun = 0.7381
    feH_sun = zz_sun/hyd_sun

    zz=zzraw.split('e-0')
    # Here is the transformation
    zz = float(zz[0]) * 10 ** -int(zz[1])
    hh = (1 - zz) * hyd
    feH = zz/hh
    feH = np.log10(feH/feH_sun)

    return feH    
    
def calc_fe_spreads(overlap, all_data, all_names, all_errors):
    """ 
    calculate how much [Fe/H] of obs match the [Fe/H] of classifications 
    """    
    feh_comp = np.zeros(len(all_names))
    gofs = np.zeros(len(all_names))
    
    spread_m = []
    spread_z = []
    for ii in range(len(all_names)):

        name = all_names[ii]
        ba_data = all_data[ii]
        try:
            ba_class_zs = overlap[name]['metallicity']
            ba_class_ms = overlap[name]['mass']            
            gofs[ii] = overlap[name]['GoFs'][0]
            
        except:
            continue

        zs0="{:.2e}".format(ba_class_zs[0])
        zs1="{:.2e}".format(ba_class_zs[1])       
        ba_class_fehmin = convert_met(zs0)
        ba_class_fehmax = convert_met(zs1)
        spread_z.append(ba_class_fehmax - ba_class_fehmin)
        spread_m.append(ba_class_ms[1] - ba_class_ms[0])

        if ba_class_fehmin <= ba_data[0] <= ba_class_fehmax:
            feh_comp[ii]=0
        elif ba_data[0] < ba_class_fehmin:
            feh_comp[ii]=ba_data[0]-ba_class_fehmin
        elif ba_data[0] > ba_class_fehmax:
            feh_comp[ii]=ba_data[0]-ba_class_fehmax

    print('mass ',np.mean(spread_m),np.std(spread_m))
    print('metallicity ',np.mean(spread_z),np.std(spread_z))

    return feh_comp, gofs  

def calc_spreads28(D_28, all_data, all_names, all_errors, names28, masses28):
    feh_comp = np.zeros(len(names28))
    gofs = np.zeros(len(names28))
    mass_comp = np.zeros(len(names28))

    spread_m = []
    spread_z = []

    for ii in range(len(names28)):

        name = names28[ii]
        mba_data = float(masses28[ii])
        for jj in range(len(all_names)):
            if all_names[jj] == name:
                ba_data = all_data[ii]
        try:
            ba_class_zs = D_28[name]['metallicity']
            ba_class_ms = D_28[name]['mass']
            gofs[ii] = D_28[name]['GoFs'][0]
            
        except:
            continue

        zs0 = "{:.2e}".format(ba_class_zs[0])
        zs1 = "{:.2e}".format(ba_class_zs[1])       
        ba_class_fehmin = convert_met(zs0)
        ba_class_fehmax = convert_met(zs1)

        spread_z.append(ba_class_fehmax - ba_class_fehmin)
        spread_m.append(ba_class_ms[1] - ba_class_ms[0])

        if ba_class_fehmin <= ba_data[0] <= ba_class_fehmax:
            feh_comp[ii] = 0
        elif ba_data[0] < ba_class_fehmin:
            feh_comp[ii] = ba_data[0] - ba_class_fehmin
        elif ba_data[0] > ba_class_fehmax:
            feh_comp[ii] = ba_data[0] - ba_class_fehmax  

        ba_class_m_min=float("{:.2f}".format(ba_class_ms[0]))
        ba_class_m_max=float("{:.2f}".format(ba_class_ms[1]))      

        if ba_class_m_min <= mba_data <= ba_class_m_max:
            mass_comp[ii] = 0
        elif mba_data < ba_class_m_min:
            mass_comp[ii] = mba_data - ba_class_m_min
        elif mba_data > ba_class_m_max:
            mass_comp[ii] = mba_data - ba_class_m_max 

    print('mass 28 ',np.mean(spread_m),np.std(spread_m))
    print('metallicity 28 ',np.mean(spread_z),np.std(spread_z))                        

    return feh_comp, gofs, mass_comp

def make_sta_bar(lab_sta_bar, overlap, option,colf,title):
    """
    make stacked bar figure to show distribution of masses in classifications
    """          

    stacked_data = []

    if option=='M':
        use_key = 'mass'
        lab = use_key +r'(M$_{\odot}$)'
        num_fig = 20
        c=1
        width = 0.20
    elif option == 'Z':
        use_key = 'metallicity'
        lab = use_key+r'(Z*10$^3$)'
        num_fig = 30
        c=0.001
        width = 0.8    

    #for each star, get data and change format
    for name in overlap.keys():
        try:
            raw_range = overlap[name][use_key]
        except:
            continue

        cleaned_range = [1 if x*c >= raw_range[0] and x*c <= raw_range[1] else 0 for x in lab_sta_bar]
  
        stacked_data.append(cleaned_range)

    plt.figure(num=num_fig)
    plt.bar(lab_sta_bar,stacked_data[0],color=colf, width=width)
    bot=np.zeros(len(lab_sta_bar))

    for ii in range(1,len(stacked_data),1):
        bot += stacked_data[ii-1]
        plt.bar(lab_sta_bar,stacked_data[ii],bottom=bot,color=colf, width=width)

    if option=='Z':
        plt.xlim(min(lab_sta_bar),max(lab_sta_bar)+1)
        plt.title(title)
    plt.xlabel(lab)
    plt.ylabel('count')   
        
def main():

    # Get file names from input
    if len(sys.argv) < 3:
        s = "Incorrect number of arguments.\n"
        s += f"Use: python3 {sys.argv[0]} <output_1.txt> ... <output_n.txt>"
        sys.exit(s)

    files = sys.argv[1:]
    stars_28 = "28stars.txt"

    # Gather all the info of the 28 stars
    names28 = []
    masses28 = []
    with open(stars_28, "r") as fread:
        for line in fread:
            name = line.split()
            names28.append(name[0])
            masses28.append(name[1])

    # Define all the directories
    dir_data = os.path.join(DIR, "Ba_star_classification_data")

    fruity_mods = "models_fruity"
    fruity_dir = os.path.join(dir_data, fruity_mods)
    models_F = get_data_fruity(fruity_dir)

    monash_mods = "models_monash"
    monash_dir = os.path.join(dir_data, monash_mods)
    models_M = get_data_monash(monash_dir)
    
    file_data = "processed_data.txt"
    file_data = os.path.join(DIR, file_data)
    all_data, all_errors, all_names, missing_values = load_ba_stars(file_data)
    
    # Uncertainty ranges in matching:
    mass_R = 0.25
    met_R = 1.7

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

    overlap, flagged_bad = find_matches(D_stars, D_range_classies,
                                        mass_R=mass_R, met_R=met_R)

    print(len(flagged_bad))
    for star in flagged_bad.keys():
        print(star)
        for key in flagged_bad[star].keys():
            print(key)
            print(flagged_bad[star][key])
        print('---')

    D_28 = {}
    D_missing =  {}
    D_rest = {}
    # divide star names into two tables: 28 and the rest
    for starname in overlap.keys():
        if starname in names28:
            D_28[starname] = overlap[starname]
        if starname in missing_values.keys() or starname in flagged_bad.keys():
            D_missing[starname] = overlap[starname]
        if starname not in names28 and starname not in missing_values.keys() and starname not in flagged_bad.keys():
            D_rest[starname] = overlap[starname]

    # Set name, label, caption of table
    table_name = 'Latex_table_28matchedstars.tex'
    table_name1 = 'Latex_table_missing_matchedstars.tex'    
    table_name2 = 'Latex_table_restmatchedstars.tex'
    table_label = 'tab:one'
    table_caption = 'caption check'

    # Write table with results
    write_matches_into_latex_table(D_28, table_name, table_label,
                                   table_caption, GoF=True)

    write_matches_into_latex_table(D_missing, table_name1, table_label,
                                   table_caption, GoF=True)   

    write_matches_into_latex_table(D_rest, table_name2, table_label,
                                   table_caption, GoF=True)
                                
    #now let's make some figures
    if 'fruity' in files[0]:
        col_figs = 'y'
        title = 'FRUITY'
    else:
        col_figs = 'b'
        title = 'MONASH'

    #histogram comparing obs and class [Fe/H]
    feh_comp28, gofs28, mass_comp28 = calc_spreads28(D_28, all_data, all_names, all_errors, names28, masses28) 
    feh_comp, gofs = calc_fe_spreads(overlap, all_data, all_names, all_errors)

    plt.figure(num=1)
    plt.hist(feh_comp28, bins=25,color=col_figs)
    plt.title(title)
    plt.xlabel(r'[Fe/H]$_{\rm{obs}}$-[Fe/H]$_{\rm{classifications}}$')
    plt.ylabel('count')

    plt.figure(num=2)
    plt.hist(mass_comp28, bins=15,color=col_figs)
    #plt.title(title)
    plt.xlabel(r'M$_{\rm{obs}}$-M$_{\rm{classifications}}$')
    plt.ylabel('count')

    plt.figure(num=3)
    plt.hist(gofs28, bins=20,color=col_figs)
    #plt.title(title)
    plt.xlabel('GoFs')
    plt.ylabel('count')

    plt.figure(num=11)
    plt.hist(feh_comp, bins=25,color=col_figs)
    plt.title(title)
    plt.xlabel(r'[Fe/H]$_{\rm{obs}}$-[Fe/H]$_{\rm{classifications}}$')
    plt.ylabel('count')

    plt.figure(num=12)
    plt.hist(gofs, bins=20,color=col_figs)
    #plt.title(title)
    plt.xlabel('GoFs')
    plt.ylabel('count')

    #make stacked bar plots with certain labels and thus alter data to fit format
    lab_sta_bar_M = [x*0.25 for x in range(4,18,1)]
    make_sta_bar(lab_sta_bar_M, overlap, 'M',col_figs,title)
    lab_sta_bar_Z = [x for x in range(1,25,1)]
    make_sta_bar(lab_sta_bar_Z, overlap, 'Z',col_figs,title)   
    plt.show()

if __name__ == "__main__":
    main()
