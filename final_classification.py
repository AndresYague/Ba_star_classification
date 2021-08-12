import numpy as np
import matplotlib.pyplot as plt
import struct, os, sys

sys.path.append("data_processing_and_plotting")
from table_maker import *
import correlation_ratios as cr

from collections import Counter

def find_matches(D_stars):
    """ finding matches between the output files of differeent algorithms.
    """

    #Initialise dictionary that holds all the matches:
    duplis = {}

    #Loop over the stars to find the matches
    for starname in D_stars.keys():

        #First get all the classifications in one list
        values = []
        for filename in D_stars[starname].keys():

            values.extend(D_stars[starname][filename])

        #Initialise the lists for the final check
        duplis[starname] = []
        only_duplis = []
        repeated = []

        #Check if value is in repeated, if yes, it's a match!
        for value in values:

            if value[0] in repeated:
               #only_duplis.append((value[0][:-1],value[1])) #without final character
               only_duplis.append((value))
            else:
               repeated.append(value[0])

        #Add duplicates to the dictionary
        duplis[starname] = only_duplis

    return duplis

def make_histogram(matches):
    '''output some statistics on the matches, including histogram'''

    nr_stars=len(matches.keys())
    matched_stars = {k:v for k,v in matches.items() if len(v) > 0}
    nr_matches=len(matched_stars.keys())

    labels=[]

    for key in matched_stars:
        for tuple_ in matched_stars[key]:
           labels.append(tuple_[0])

    counted_ls = Counter(sorted(labels))
    x_axis = counted_ls.keys()
    y_axis = counted_ls.values()
    plt.xticks(range(len(x_axis)), x_axis, rotation=60, fontsize=9)
    plt.bar(x_axis,y_axis)
    plt.tight_layout()
    plt.savefig('histo.pdf')
    plt.show()

def write_matches_to_txt(matches):
    """create .txt file with all matches so it can be used in plotting scripts
    """
    for key in matches:
        if len(matches[key]) != 0:
           #print('------')
           print("For star {}:".format(key))
           for tuple_ in matches[key]:
              lab=name_check_reverse(tuple_[0])
              # Print
              s = f"Label {lab} with goodness of fit {tuple_[1]}"
              s += f" and dilution {tuple_[2]} average residual {tuple_[3]}"
              print(s)
           print("------")

def main():
    """Load .txt files with output of classification algorithms, including
    goodness of fit. Compare the outputs per star: 1)do the algorithms agree?
    2)is GoF above the threshold? 3)make histogram with GoF values 4)make .tex
    table with all outcomes? """

    #Get file names from input
    if len(sys.argv) < 2:
       s = "Incorrect number of arguments. "
       s += f"Use: python3 {sys.argv[0]} {txt} {txt} "
       raise Exception(s)

    files=sys.argv[1:]

    #Make dictionaries with all the data
    D_files, D_stars = read_files_into_dicts(files) #read_txt_into_dicts(files)

    matches = find_matches(D_stars)

    write_matches_to_txt(matches)

    make_histogram(matches)

    #Set name, label, caption of table
    table_name='Latex_table_matchedstars.tex'
    table_label='tab:one'
    table_caption='caption check'

    #Write table with results
    write_matches_into_latex_table(matches,table_name,table_label,table_caption,GoF=True)

    #Set name, label, caption of table
    table_name='Latex_table_GoF.tex'
    table_label='tab:one'
    table_caption='caption check'

    #Write table with results
    write_into_latex_table(D_stars,table_name,table_label,table_caption,GoF=True)
    create_names_table('names.tex')

if __name__ == "__main__":
    main()
