import os, glob, random
import numpy as np

def apply_dilution(elems, kk, names, zero = 0):
    """
    Apply dilution of kk. This formula only works for heavy elements

    kk is the fraction of the Ba-star envelope that comes from the AGB
    """

    # Just apply the formula to each element in name
    new_elements = {}
    for name in names:
        try:
            new_elements[name] = np.log10((1 - kk) + kk * 10 ** elems[name])
        except KeyError:
            continue
        except:
            raise

    # It may happen that all the abundances are very low, so skip those cases
    all_zero = True
    for name in names:
        try:
            if abs(new_elements[name]) > zero:
                all_zero = False
                break
        except KeyError:
            continue
        except:
            raise

    if all_zero:
        new_elements = None

    return new_elements

def convert_metallicity(zz):
    """
    Convert metallicity to feH, zz is a string
    """

    # Define several quantities to use
    zz_sun = 0.014
    hyd = 0.75
    hyd_sun = 0.7381
    feH_sun = zz_sun/hyd_sun

    # Careful if "sun" in name
    if zz == "sun":
        feH = 0
    else:
        # Here is the transformation
        zz = int(zz[0]) * 10 ** -int(zz[-1])
        hh = (1 - zz) * hyd
        feH = zz/hh
        feH = np.log10(feH/feH_sun)

    return feH

def get_string_names(feH, elems, names, label):
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
        if name == "Fe/H":
            val = feH
        else:
            val = elems[name]

        # Be careful with undefined values
        if val == "-":
            ss += " -"
        else:
            # Add to the string
            ss += " {:5.2f}".format(val)

    # Add newline
    ss += " " + label + "\n"

    return ss

def process_fruity(directory, outpt, names, zero = 0):
    """
    Process fruity data to extract the elements we want
    """

    # Generate the wildcard for the models
    wildcard = os.path.join(directory, "*")
    model_files = glob.glob(wildcard)
    for model_file in model_files:
        # First extract the label
        file_name = os.path.split(model_file)[-1].split(".")[0]
        label = "fruity_" + file_name

        # Now get the Fe/H from the file_name
        indx_z = file_name.index("z")
        zz = file_name[indx_z + 1: indx_z + 4]

        # Convert metallicity to Fe/H
        feH = convert_metallicity(zz)

        # Now for each line in the model file, retrieve the abundances
        with open(model_file, "r") as fread:
            fread.readline() # Skip first line

            # Dictionary for abundances
            elems = {}
            for line in fread:
                lnlst = line.split()

                # Add iron to the name
                lnlst[0] += "/Fe"

                # Take last element
                try:
                    elems[lnlst[0]] = float(lnlst[-1])
                except ValueError:
                    elems[lnlst[0]] = "-"
                except:
                    raise

        # Apply dilutions from kk = 0 to kk = 1 with n steps
        dk = 0.01
        kk_arr = np.arange(0, 1 + dk, dk)
        for kk in kk_arr:

            # Calculate dilution
            new_elements = apply_dilution(elems, kk, names, zero = zero)

            # Ignore too much dilution
            if new_elements is None:
                continue

            # Now add the relevant elements to the string
            ss = get_string_names(feH, new_elements, names, label)

            # Now write this string
            with open(outpt, "a") as fappend:
                fappend.write(ss)

def process_monash(directory, outpt, names, zero = 0):
    """
    Process fruity data to extract the elements we want
    """

    # Generate the wildcard for the models
    wildcard = os.path.join(directory, "*")
    model_files = glob.glob(wildcard)
    for model_file in model_files:
        # First extract the filename
        file_name = os.path.split(model_file)[-1].split(".")[0]

        # Now for each line in the model file, retrieve the abundances
        with open(model_file, "r") as fread:
            # There are several models in each file, so just
            # look for each and save it
            for line in fread:
                # Find initial mass
                if "Initial mass" in line:
                    # Get label
                    lnlst = line.split()

                    # Initial mass
                    ini_mass = lnlst[4][0:4]
                    label = "monash_m" + ini_mass + "_"

                    # Mix
                    mix = lnlst[13][0:8]
                    label += "mix_" + mix + "_"

                    # Ov
                    if "_ov" in line:
                        ov = lnlst[16]
                        label += "N_ov_" + ov + "_"

                    # Rest of the label
                    label += file_name.split("_")[-1]

                # Now model
                if "Final abundances" in line:
                    fread.readline() # Skip header

                    # Save elements
                    elems = {}
                    for line in fread:
                        if "#" in line:
                            break

                        # Add element to the list
                        lnlst = line.split()
                        name = lnlst[0].capitalize()
                        if name == "Fe":
                            feH = float(lnlst[3])
                        else:
                            name += "/Fe"
                            elems[name] = float(lnlst[4])

                    # Apply dilutions from kk = 0 to kk = 1 with difference dk
                    dk = 0.01
                    kk_arr = np.arange(0, 1 + dk, dk)
                    for kk in kk_arr:

                        # Calculate dilution
                        new_elements = apply_dilution(elems, kk, names,
                                                      zero = zero)

                        # Ignore too much dilution
                        if new_elements is None:
                            continue

                        # Now add the relevant elements to the string
                        ss = get_string_names(feH, new_elements, names, label)

                        # Now write this string
                        with open(outpt, "a") as fappend:
                            fappend.write(ss)

def eliminate_same_models(processed_models):
    """
    Read all models and eliminate repeated ones
    """

    with open(processed_models, "r") as fread:
        with open("temp.txt", "w") as fwrite:
            prev_Mod = None
            for line in fread:
                if prev_Mod is None:
                    prev_Mod = line
                    fwrite.write(line)
                    continue

                if line != prev_Mod:
                    fwrite.write(line)

                prev_Mod = line

    os.rename("temp.txt", processed_models)

def shuffle_models(processed_models, shuffled_models):
    """
    Shuffle the models
    """

    # Load models to memory
    all_models = []
    with open(processed_models, "r") as fread:
        header = fread.readline()
        for line in fread:
            all_models.append(line)

    # Now shuffle them
    random.shuffle(all_models)

    # And write them
    with open(shuffled_models, "w") as fwrite:
        fwrite.write(header)
        for line in all_models:
            fwrite.write(line)

def process_data(data_file, processed_data, names):
    """
    Get the information out for all the Ba stars
    """

    with open(data_file, "r") as fread:

        # Header
        header = fread.readline().split()

        # Rest
        all_vals = []
        for line in fread:
            lnlst = line.split()

            # Add new dictionary
            all_vals.append({})
            all_vals[-1]["name"] = lnlst[0]

            # For each name, get the value and error
            for name in names:

                # First the easy ones
                spltName = name.split("/")[0]
                if name in header:
                    indx = header.index(name)
                    indx_err = indx + 1
                elif spltName in header:
                    indx = header.index(spltName)
                    indx_err = header.index("err_" + spltName)

                # Introduce values
                try:
                    all_vals[-1][name] = float(lnlst[indx])

                    # If error is zero, just put 0.5
                    val_err = float(lnlst[indx_err])
                    if val_err < 1e-2:
                        val_err = 0.5

                    all_vals[-1][name + "_err"] = val_err
                except ValueError:
                    all_vals[-1][name] = "-"
                    all_vals[-1][name + "_err"] = "-"
                except:
                    raise

    # Now just write all the data
    with open(processed_data, "w") as fwrite:

        # First the header
        header = "#"
        for name in names:
            header += " {0} {0}_err".format(name)
        header += " Name\n"

        fwrite.write(header)

        # Now the values and errors
        for dic in all_vals:
            s = ""
            for name in names:
                try:
                    s += " {:5.2f} {:5.2f}    ".format(dic[name], dic[name + "_err"])
                except ValueError:
                    s += "   -     -      "

            s += "{}\n".format(dic["name"])
            fwrite.write(s)

def main():
    """
    Just process all the data to create consistent models
    """

    # Names of quantities we want
    # Element set 1
    names = [
             "Fe/H", "Rb/Fe", "Sr/Fe", "Zr/Fe", "Mo/Fe", "Ru/Fe"
             "La/Fe", "Ce/Fe", "Nd/Fe", "Eu/Fe"
            ]

    # Element set 2
    names = [
             "Fe/H", "Rb/Fe", "Sr/Fe", "Zr/Fe", "Y/Fe",
             "La/Fe", "Ce/Fe", "Nd/Fe", "Eu/Fe"
            ]

    # Define all the directories
    dir_data = "Ba_star_classification_data"
    fruity_mods = "models_fruity"
    monash_mods = "models_monash"
    data_file = "all_abund_and_masses.dat"

    fruity_dir = os.path.join(dir_data, fruity_mods)
    monash_dir = os.path.join(dir_data, monash_mods)
    data_file = os.path.join(dir_data, data_file)

    # Names for output files
    processed_models = "processed_models.txt"
    shuffled_models = "shuffled_models.txt"
    processed_data = "processed_data.txt"

    # Write the header
    header = "# " + " ".join(names) + " Label " + "\n"
    with open(processed_models, "w") as fwrite:
        fwrite.write(header)

    # Process fruity
    process_fruity(fruity_dir, processed_models, names, zero = 0.2)

    # Process monash
    #process_monash(monash_dir, processed_models, names, zero = 0.2)

    # Check that all models are different enough
    print("Eliminating same models...")
    eliminate_same_models(processed_models)

    # Now shuffle the models
    print("Shuffling models")
    shuffle_models(processed_models, shuffled_models)

    # Process data
    process_data(data_file, processed_data, names)

if __name__ == "__main__":
    main()
