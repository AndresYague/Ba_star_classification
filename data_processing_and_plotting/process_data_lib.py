import os, glob
import string
import numpy as np

ZERO_ERROR = 0.5
MIN_VAL_ERR = 1e-2

def short_name_generator(name, shortnames=None):
    '''
    Generate a short name from a long name
    '''

    # Break down the name
    split_name = name.split("_")

    # Fruity or monash?
    if "fruity" in name:

        # Break down the name further
        split_name = split_name[1].split("z")

        # Get the mass
        mass = split_name[0].replace("p", ".")

        # Get the metallicity
        metallicity = "z"
        try:
            metallicity += "0" * (int(split_name[1][-1]) - 1)
            metallicity += split_name[1][0]
        except ValueError:
            metallicity += "014"
        except:
            raise

        # Final short name
        short = f"F-{mass}{metallicity}"

    elif "monash" in name:

        # Mass and metallicity
        mass = split_name[1]
        metallicity = split_name[-1]

        short = f"M-{mass}{metallicity}"

    else:
        raise Exception("Cannot transform this name")

    # Do not give letter if there's no list
    if shortnames is None:
        return short

    # Put the letter at the end
    for letter in string.ascii_lowercase:
        if short + letter not in shortnames:
            return short + letter

def new_names(dir_=None):
    '''Fills the name-lists'''

    fullnames = []
    shortnames = []

    if dir_ is not None:
        path_file = os.path.join(dir_, "all_names.txt")
    else:
        path_file = "all_names.txt"

    if os.path.isfile(path_file):

        # Load names from all_names.txt file
        with open(path_file, "r") as fread:
            for line in fread:
                name, short = line.split()

                fullnames.append(name)
                shortnames.append(short)

    else:

        files = ["processed_models_fruity.txt",
                 "processed_models_monash.txt"]

        for file_ in files:
            with open(file_, "r") as fread:
                # Skip header
                next(fread)

                # Get names
                for line in fread:
                    name = line.split()[-1]

                    # Put names if not there
                    if name not in fullnames:
                        short = short_name_generator(name, shortnames)

                        fullnames.append(name)
                        shortnames.append(short)

        # Save names in all_names.txt file
        with open(path_file, "w") as fwrite:
            for name, short in zip(fullnames, shortnames):
                fwrite.write(f"{name} {short}\n")

    return fullnames, shortnames

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

def get_data_values(data_file, names=None):
    """
    Get the information out for all the Ba stars
    """

    with open(data_file, "r") as fread:

        # Header
        header = fread.readline().split()

        # Construct a names list if not provided
        if names is None:
            names = []
            for elem in header[1:-4]:
                if "err" not in elem:
                    if "/" in elem:
                        names.append(elem)
                    else:
                        names.append(elem + "/Fe")

        # Rest
        all_vals = {}
        for line in fread:
            lnlst = line.split()

            # Add new dictionary
            star_name = lnlst[0]
            all_vals[star_name] = {}

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
                    all_vals[star_name][name] = float(lnlst[indx])

                    # If error is zero or "-", just put ZERO_ERROR
                    try:
                        val_err = float(lnlst[indx_err])
                        if val_err < MIN_VAL_ERR:
                            val_err = ZERO_ERROR
                    except ValueError:
                        val_err = ZERO_ERROR
                    except:
                        raise

                    all_vals[star_name][name + "_err"] = val_err
                except ValueError:
                    all_vals[star_name][name] = "-"
                    all_vals[star_name][name + "_err"] = "-"
                except:
                    raise

    return all_vals

def get_data_fruity(directory):
    """
    Get the fruity data in a dictionary
    """

    # Generate the wildcard for the models
    wildcard = os.path.join(directory, "*")
    model_files = glob.glob(wildcard)
    all_models = {}
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

        # Store
        all_models[label] = elems
        all_models[label]["Fe/H"] = feH

    return all_models

def get_data_monash(directory):
    """
    Get the monash data in a dictionary
    """

    # Generate the wildcard for the models
    wildcard = os.path.join(directory, "*")
    model_files = glob.glob(wildcard)
    all_models = {}
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

                    # Store
                    all_models[label] = elems
                    all_models[label]["Fe/H"] = feH

    return all_models

def apply_dilution(elems, kk, names, zero=0):
    """
    Apply dilution of kk. This formula only works properly for heavy elements

    kk is the fraction of the Ba-star envelope that comes from the AGB
    """

    # Just apply the formula to each element in name
    new_elements = {}
    for name in elems:

        # Skip metallicity
        if (name in names) and (name != "Fe/H"):
            try:
                new_elements[name] = np.log10((1 - kk) + kk * 10 ** elems[name])
            except TypeError:
                new_elements[name] = elems[name]
            except:
                raise
        else:
            new_elements[name] = elems[name]

    # It may happen that all the abundances are very low, so skip those cases
    all_zero = True
    for name in names:

        # Skip metallicity
        if name == "Fe/H":
            continue

        if abs(new_elements[name]) > zero:
            all_zero = False
            break

    if all_zero:
        new_elements = None

    return new_elements
