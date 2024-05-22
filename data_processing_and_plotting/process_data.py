import os, glob, sys
import numpy as np
from process_data_lib import *

DK_STEP = 0.002
ZERO = 0.2

def process_data(data_file, output_file, names):
    """
    Get the information out for all the Ba stars and into output_file
    """

    # Get values
    all_vals = get_data_values(data_file, names=names)

    # Now just write all the data
    if output_file is not None:
        with open(output_file, "w") as fwrite:

            # First the header
            header = "#"
            for name in names:
                header += f" {name} {name}_err"
            header += " Name\n"

            fwrite.write(header)

            # Now the values and errors
            for star_name in all_vals:

                # Get the dictionary
                dic = all_vals[star_name]

                # Format
                s = ""
                for name in names:
                    try:
                        name_err = name + "_err"
                        s += f" {dic[name]:6.3f} {dic[name_err]:6.3f}    "
                    except ValueError:
                        s += "   -     -      "

                s += f"{star_name}\n"
                fwrite.write(s)

def get_string_names(elems, names, label):
    """
    Transform the list of values into a string to write
    elems is a dictionary with the values
    names is the selected keys we want for the output
    label is the model label
    """

    # Start to write the output string
    ss = ""
    for name in names:

        # Get back the value
        val = elems[name]

        # Be careful with undefined values
        if val == "-":
            ss += " -"
        else:
            # Add to the string
            ss += f" {val:6.3f}"

    # Add newline
    ss += " " + label + "\n"

    return ss

def process_models(directory, outpt, names, zero=0, with_dilution=True):
    """
    Process fruity and monash data to extract the elements we want
    """

    # Choose correct models
    if "fruity" in directory:
        all_models = get_data_fruity(directory)
    elif "monash" in directory:
        all_models = get_data_monash(directory)
    else:
        raise Exception("Only implemented for fruity or monash models")

    for label in all_models:
        # Ignore T60 label
        if "T60" in label:
            continue
        elems = all_models[label]

        # If calculating for the NN, then shorten the label
        #if with_dilution:
        #    label = short_name_generator(label)

        # Apply dilutions from kk = 0 to kk = 1 with DK_STEP size
        kk_arr = np.arange(0, 1 + DK_STEP, DK_STEP)
        for kk in kk_arr:

            # Calculate dilution
            if with_dilution or kk == kk_arr[-1]:
                new_elements = apply_dilution(elems, kk, names, zero=zero)
            else:
                continue

            # Ignore too much dilution
            if new_elements is None:
                continue

            # Now add the relevant elements to the string
            ss = get_string_names(new_elements, names, label)

            # Now write this string
            with open(outpt, "a") as fappend:
                fappend.write(ss)

def eliminate_same_models(processed_models):
    """
    Read all models and eliminate repeated ones
    """

    # Organize the models by label first
    label_models = dict()
    with open(processed_models, "r") as fread:
        header = fread.readline()

        # For each line separate label and model
        for line in fread:
            lnlst = line.split()


            # Extract label and model
            label = lnlst[-1]
            model = " ".join(lnlst[:-1])

            # Add to dictionary
            label_models[label] = label_models.get(label, []) + [model]

    # Now write them
    with open("temp.txt", "w") as fwrite:
        fwrite.write(header)

        for label in label_models:
            # Eliminate repeated
            set_of_models = set(label_models[label])

            # And write
            for model in set_of_models:
                s = f"{model} {label}\n"
                fwrite.write(s)

    os.rename("temp.txt", processed_models)

def main():
    """
    Just process all the data to create consistent models
    """

    s = f'Use: python3 {sys.argv[0]} [y/n]\n'
    s += 'Where "y" indicates the use of dilution for the models.\n'
    s += 'Default option is y.\n'
    print(s)

    with_dilution = True
    if len(sys.argv) > 1:
        with_dilution = True if  sys.argv[1] == "y" else False

    s = "Processing data "
    if with_dilution:
        s += "with "
    else:
        s += "without "
    s += "dilution\n"
    print(s)

    # Load element set
    with open(os.path.join(sys.path[0],"element_set.dat"), "r") as fread:
        for line in fread:
            lnlst = line.split()

            # Skip comments and empty lines
            if len(lnlst) == 0 or "#" in lnlst[0]:
                continue

            names = lnlst
            break

    # Echo element list
    print(f"Element list: {' '.join(names)}")

    # Define all the directories
    dir_data = "~/Ba_star_classification_VB/Ba_star_classification_data"
    fruity_mods = "models_fruity_dec"
    monash_mods = "models_monash"
    #data_file = "all_abund_and_masses.dat"
    data_file = "all_data_w_err.dat"

    fruity_dir = os.path.join(dir_data, fruity_mods)
    monash_dir = os.path.join(dir_data, monash_mods)
    data_file = os.path.join(dir_data, data_file)

    # Names for output files
    processed_models_fruity = os.path.join(sys.path[0], "processed_models_fruity.txt")
    processed_models_monash = os.path.join(sys.path[0], "processed_models_monash.txt")
    processed_data = os.path.join(sys.path[0], "processed_data.txt")

    # Process data
    print("Processing observations...")
    process_data(data_file, processed_data, names)

    # Write the header
    header = "# " + " ".join(names) + " Label " + "\n"

    # Process fruity
    print("Processing fruity...")
    with open(processed_models_fruity, "w") as fwrite:
        fwrite.write(header)
    process_models(fruity_dir, processed_models_fruity, names, zero=ZERO,
                    with_dilution=with_dilution)

    # Process monash
    print("Processing monash...")
    with open(processed_models_monash, "w") as fwrite:
        fwrite.write(header)
    process_models(monash_dir, processed_models_monash, names, zero=ZERO,
                    with_dilution=with_dilution)

    # Check that all models are different enough
    if with_dilution:
        print("Eliminating same models...")
        eliminate_same_models(processed_models_fruity)
        eliminate_same_models(processed_models_monash)

    print("Done!")

if __name__ == "__main__":
    main()