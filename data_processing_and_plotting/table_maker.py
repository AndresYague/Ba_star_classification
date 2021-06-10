import os, glob, re, sys
import string
import numpy as np
from process_data_lib import *

def name_check(name):
    '''This function rrenames the models with a short name'''

    #Load all the names
    fulln,shortn=new_names()

    #Check if name is included in full list, if it is,
    #replace it with the short name and return the replacement
    for i in range(len(fulln)):

        if name == fulln[i]:

           name = shortn[i]

    return(name)

def create_names_table(filename):
    '''Creates latex table with full and shortnames'''

    #Load all the names
    fnames,snames=new_names()

    #Sorting the names alphabetically
    full_list_di={}
    for i in range(len(fnames)):
        full_list_di[snames[i]] = fnames[i]

    items = full_list_di.items()
    sorted_items = sorted(items)

    #Creation of latex table
    #Step 1: Open file and write header:
    g=open(filename,'w')

    #Step 2: Define caption and label
    tab_cap = 'This table lists the full and short names of the AGB models.'
    tab_lab = 'tab:names'

    #Step 3: Write headings etc to table
    table_def = '\\begin{longtable}{ll}\n\\centering\n\\caption{'
    table_def += tab_cap + '}\\label{' + tab_lab + '}\\\ \n'
    g.write(table_def)

    tab_headings='Full name & Short name \\\ \n'
    g.write(tab_headings)
    g.write('\\hline\n')

    #Step 4: Write the two sets of names
    for key,val in sorted_items:
        val=val.replace('_','\_')
        g.write(key+' & '+val+'\\\ \n')

    #Step 5: Write the final latex commands
    g.write('\\hline\n')
    table_end=('\\end{longtable}')
    g.write(table_end)
    g.close()

def get_clean_lnlst(line):
    '''Clean the input by removed unneeded words'''

    lnlst=line.split()
    if 'star ' in line:
       return(lnlst)
    elif 'Label' in line:
       name=name_check(lnlst[1])
       return [ name,lnlst[5],lnlst[7],lnlst[10]]
    else:
       return(None)

def get_clean_filename(filen):
    '''Clean the filename by removing unneeded words'''

    filen = filen.replace('_correlated','').replace('tensorflow_','NN-')
    filen = filen.replace('.txt','').replace('_','-').replace('clo','Clo')
    filen = filen.replace('diluted','Dil')

    return(filen)

def read_files_into_dicts(files):
    '''Find all result files and load them'''

    # Initialize dictionary
    dict_files = {}
    # Keep an inventory of repeated models
    repeated = {}

    for model_file in files:

        # First extract the filename
        fname = os.path.split(model_file)[-1]
        file_name=get_clean_filename(fname)
        type_=file_name

        # Initialize the 2D dictionaries
        dict_files[type_] = {}
        repeated[type_] = {}

        # For each star in the file, save the matched model
        with open(model_file,"r") as fread:
             for line in fread:
                 lnlst=get_clean_lnlst(line)
                 if lnlst is None:
                    continue

                 #Read line, which has either star name or matched model
                 if 'star' in lnlst:
                    star_name = lnlst[-1][:-1]
                    if star_name not in dict_files:

                        # The starname set
                        dict_files[type_][star_name] = []

                        # Add the list to the repeated models
                        repeated[type_][star_name] = []

                # Add this line to the dictionary
                 else:
                    # Check if repeated to skip
                    if lnlst[0] in repeated[type_][star_name]:
                        continue

                    # Add this model here to avoid repeating it
                    repeated[type_][star_name].append(lnlst[0])

                    # Add to the set
                    dict_files[type_][star_name].append(tuple(lnlst))

    #Make another dictionary, sorted per star instead of file
    star_dict={}

    #Loop over dictionary to create dictionary framework
    for type_ in dict_files.keys():

        for star_name in dict_files[type_].keys():

            #Initiate new dictionary with layers
            star_dict[star_name]= {}
            star_dict[star_name][type_] = {}

    #Fill new dictionary
    for type_ in dict_files.keys():
        for star_name in dict_files[type_].keys():

            star_dict[star_name][type_]=dict_files[type_][star_name]

    return dict_files, star_dict

def clean_match(match):
    """ Turn match into latex format
    """

    match=match.replace('(','').replace(')','').replace('\'','').replace(',','&')
    match=match.replace('%','\%')

    return(str(match))

def write_into_latex_table(star_di,tab_name,tab_label,tab_caption):
    '''All results turned into a compilable
    latex tables'''

    #Filter the matches, so that we only print either fruity or monash matches
    #Step 1: define what to remove

    for starname in star_di.keys():

        for filename in star_di[starname].keys():
            if 'fruity' in filename:
               rem_name = 'M-'
               break
            elif 'monash' in filename:
               rem_name = 'F-'
               break
            else:
               rem_name = 'Q-'

    #Step 2: do the removal
    for starname in star_di.keys():

        for filename in star_di[starname].keys():

            values = star_di[starname][filename]

            if len(values) > 0:
               while len(values) > 0 and rem_name in values[0][0]:
                  star_di[starname][filename].remove(values[0])

            if len(values) > 1:
               while rem_name in values[-1][0]:
                  star_di[starname][filename].remove(values[-1])



    #Empty array to add when a file doesn't have as many matches as other files
    empty=(' ',' ',' ',' ')

    #Keeping track of file names (needed for table headings)
    list_files=[]

    #Make the list of matches equal length for all files
    for starname in star_di.keys():

        lens=np.zeros(len(star_di[starname]))
        i=0
        for filename in star_di[starname].keys():
            #Fill array 'lens' in lengths of matches
            lens[i]=len(star_di[starname][filename])

            if len(list_files) < len(star_di[starname]):
               list_files.append(filename)
            i+=1

        #Check is lens[i] is smaller than max(lens), and add matches if it is
        for filename in star_di[starname].keys():

            while len(star_di[starname][filename]) < max(lens):

                  star_di[starname][filename].append(empty)

    #Open file and write header:
    g=open(tab_name,'w')

    num_columns=len(star_di[starname].keys())
    cols=4 #labels, prob, dil, res

    table_def='\\begin{longtable}{c'
    for i in range(num_columns):
        table_def += '|l'+'c'*(cols-1)
    table_def += '} \\caption{'+tab_caption+'}\\label{'+tab_label+'}\\\ \n'
    g.write(table_def)

    list_files=sorted(list_files)

    table_labels='Star '
    for i in range(len(list_files)):
        table_labels += ' & \\multicolumn{'+str(cols)+'}{|l}{'+clean_match(list_files[i])+'}'
    table_labels += '\\\ \n'
    g.write(table_labels)

    table_headings=' & Label & Prob & Dil & Res'*len(list_files)+'\\\ \n'

    g.write(table_headings)
    g.write('\\hline\n')

    #Write matches to file:
    for starname in star_di.keys():

        g.write(starname)
        an_array=[]
        for i in list_files:
            an_array.append(star_di[starname][i])

        for i in range(len(an_array[0])):

            for j in range(len(an_array)):
                 cl_ma=clean_match(str(an_array[j][i]))
                 g.write('&'+cl_ma)

            g.write('\\\ \n')
        g.write('\\hline\n')
        #stop
    g.write('\\hline\n')

    table_end=('\\end{longtable}')
    g.write(table_end)

def main():
    """
    Load files with classification results and save as latex table
    """

    #Get file names from input
    if len(sys.argv) < 2:
       s = "Incorrect number of arguments. "
       s += f"Use: python3 {sys.argv[0]} <file1> [file2 ...]"
       sys.exit(s)

    files = sys.argv[1:]

    #Make dictionaries with all the data
    dict_models,dict_stars = read_files_into_dicts(files)

    #Make table with long and short names
    create_names_table('names.tex')

    #Set name, label, caption of table
    table_name='Latex_table_results.tex'
    table_label='tab:one'
    table_caption='caption check'

    #Write table with results
    write_into_latex_table(dict_stars,table_name,table_label,table_caption)


if __name__ == "__main__":
    main()
