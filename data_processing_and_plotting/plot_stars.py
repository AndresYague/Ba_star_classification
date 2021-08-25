import sys, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from process_data_lib import *

FIGSIZE = [20, 8]
EDGECOLOR = "midnightblue"
MARKERCOLOR = "yellow"

FONTSIZE = 18
MARKERSIZE = 16
LINESIZE = 2

ZMIN = 6
IRONZ = 26
LIMIT_DIL = False

def get_dict_predicted(files):
    """
    Make input dictionary to combine all files
    """

    # Get the short names
    fullnames, shortnames = new_names()
    short_names_dict = {full: short for full, short in
                                        zip(fullnames, shortnames)}
    full_names_dict = {short: full for short, full in
                                        zip(shortnames, fullnames)}

    # Initialize dictionary
    dict_ = {}
    dict_["fruity"] = {}
    dict_["monash"] = {}

    # Keep an inventory of repeated models
    repeated = {}
    repeated["fruity"] = {}
    repeated["monash"] = {}

    # Go for file and line
    for file_ in files:
        with open(file_, "r") as fread:

            for line in fread:
                lnlst = get_clean_lnlst(line)

                # Skipe lines without content
                if lnlst is None:
                    continue

                # Read the star name and add the sets in
                # the dictionaries if they were not there
                if "star" in lnlst:
                    star_name = lnlst[-1][:-1]
                    if star_name not in dict_["fruity"]:

                        # The starname set
                        dict_["fruity"][star_name] = set()
                        dict_["monash"][star_name] = set()

                        # Add the list to the repeated models
                        repeated["fruity"][star_name] = []
                        repeated["monash"][star_name] = []

                # Add this line in fruity or monash
                else:

                    lnlst[0] = full_names_dict[lnlst[0]]
                    if "fruity" in lnlst[0]:
                        type_ = "fruity"
                    elif "monash" in lnlst[0]:
                        type_ = "monash"

                    # Check if repeated to skip
                    if lnlst[0] in repeated[type_][star_name]:
                        continue

                    # Add this model here to avoid repeating it
                    repeated[type_][star_name].append(lnlst[0])

                    # Add to the set
                    lnlst[1] = lnlst[1]
                    dict_[type_][star_name].add(tuple(lnlst))

    return dict_

def plot_this_data(data, name_z, ax1, ax2, fmt, fmtMk="", label=None, mec=None,
                   mfc=None, error=False, data_compare=None):
    '''
    Plot a specific diluted model
    '''

    # Order data
    values = [[]]
    x_axis = [[]]
    vals_err = [[]]
    for z in name_z:

        # This part prepares the keys to be looked for in data
        name = name_z[z]
        key = name + "/Fe"
        if error:
            key_err = key + "_err"

        # Here the values are retrieved, giving the 0.5 errorbar if not
        # present
        if key in data:
            val = data[key]
            if error:
                val_err = data[key_err]
                if val_err == "-":
                    val_err = 0.5

            # If the value does not exist, leave a space
            if val == "-":
                x_axis.append([])
                values.append([])
                vals_err.append([])
            else:
                x_axis[-1].append(z)
                values[-1].append(val)
                if error:
                    vals_err[-1].append(val_err)
                else:
                    vals_err[-1].append(0)


    # Plot the model
    color = None
    for x_line, y_line, y_err in zip(x_axis, values, vals_err):

        # Repeating arguments
        args = {"mec": mec, "ms": MARKERSIZE, "lw": LINESIZE, "mfc": mfc}
        if not error:
            args["ms"] /= 3

        # Plot
        if not error:

            # Plotting model
            if color is None:
                line = ax1.plot(x_line, y_line, fmtMk + fmt, label=label,
                                **args)
                color = line[-1].get_color()
            else:
                ax1.plot(x_line, y_line, fmtMk + fmt, color=color, **args)
        else:

            # Plotting data
            ax1.errorbar(x_line, y_line, yerr=y_err, fmt=fmtMk + fmt,
                         ecolor=mec, label=label, capsize=3, zorder=5, **args)
            label = None

    # Plot residuals
    if data_compare is not None:

        # Dictionaries for keys so that they are easy to compare
        names_to_z = {name_z[z] + "/Fe": z for z in name_z}
        names_Fe = names_to_z.keys()

        # Make sure that it's the data we want to compare
        for name in data_compare:
            if name in names_Fe and name in data:

                # Positive residuals only
                try:
                    residual = data_compare[name] - data[name]
                except TypeError:
                    continue
                except:
                    raise

                # Plot it, reducing a bit the markersize with respect to data
                ax2.plot(names_to_z[name], residual, fmtMk, color=color,
                         ms=MARKERSIZE / 2)

def plot_results(predicted_models_dict, fruity_models_dict,
                 monash_models_dict, dict_data, red_elements, pathn):
    """
    Plot the results, star by star.
    """

    # Get the short names
    fullnames, shortnames = new_names()
    short_names_dict = {full: short for full, short in
                                        zip(fullnames, shortnames)}
    full_names_dict = {short: full for short, full in
                                        zip(shortnames, fullnames)}

    # Grab the data
    name_z = np.loadtxt(os.path.join("data_for_plot", "atomic_nums.dat"),
                        dtype=str)

    # Create dictionaries
    red_elements = {int(z): name for name, z in name_z
                if name in red_elements}
    name_z = {int(z): name for name, z in name_z
                if name != "element" and int(z) >= ZMIN}

    # Create names list for dilution.
    names_dil = []
    for z in name_z:
        name = name_z[z]

        # To limit to anything above Fe
        if LIMIT_DIL and z > IRONZ:
            names_dil.append(name + "/Fe")
        elif not LIMIT_DIL:
            names_dil.append(name + "/Fe")

    # Each key in dict_data is a star
    for key in dict_data:

        # Figure specification
        fig = plt.figure(figsize=FIGSIZE)
        spec = gridspec.GridSpec(8, 1)

        # Axes for abundances
        ax1 = plt.subplot(spec[:4, :])
        ax1.set_title(key, size=FONTSIZE * 1.5)
        ax1.set_ylabel("[X/Fe]", size=FONTSIZE)

        # Axes for residual
        ax2 = plt.subplot(spec[4:6, :], sharex=ax1)
        ax2.set_ylabel("residuals", size=FONTSIZE)

        # Remove vertical space between plots
        fig.subplots_adjust(hspace=0)

        # Plot the fruity and monash models
        mod_type = ["fruity", "monash"]
        n_plots = 0
        for type_ in mod_type:

            # Retrieve name and dilution
            for model_name, dil in predicted_models_dict[type_][key]:

                # Get model and dilute
                if type_ == "fruity":
                    model = fruity_models_dict[model_name]
                    fmt = "-"
                    fmtMk = "v"
                elif type_ == "monash":
                    model = monash_models_dict[model_name]
                    fmt = "--"
                    fmtMk = "o"
                else:
                    raise Exception("Only types implemented: fruity and monash")

                diluted_model = apply_dilution(model, dil, names_dil)

                # plot
                n_plots += 1
                short_name = short_names_dict[model_name]
                plot_this_data(diluted_model, name_z, ax1, ax2,
                               label=short_name, fmt=fmt, fmtMk=fmtMk,
                               data_compare=dict_data[key])

        # Plot data and errorbars
        plot_this_data(dict_data[key], name_z, ax1, ax2, label="Data",
                        fmt="*", mec=EDGECOLOR, mfc=MARKERCOLOR, error=True)

        # Red elements
        plot_this_data(dict_data[key], red_elements, ax1, ax2,
                        fmt="*", mec=EDGECOLOR, mfc="red", error=True)

        # set vertical lines
        for ii in range(ZMIN, max(name_z.keys()) + 1, 4):
            ax1.axvline(ii, ls="-", color="lightgray", zorder=0)
            ax2.axvline(ii, ls="-", color="lightgray", zorder=0)
        for ii in range(ZMIN + 2, max(name_z.keys()) + 1, 4):
            ax1.axvline(ii, ls="--", color="lightgray", zorder=0)
            ax2.axvline(ii, ls="--", color="lightgray", zorder=0)

        # Set horizontal lines
        ax1.axhline(ls="--", color="silver", zorder=0)
        ax2.axhline(ls="-", color="k", zorder=0)
        ax2.axhline(0.2, ls="--", color="k", zorder=0)
        ax2.axhline(-0.2, ls="--", color="k", zorder=0)
        ax2.axhline(0.4, ls=":", color="k", zorder=0)
        ax2.axhline(-0.4, ls=":", color="k", zorder=0)

        # Adjust axes
        ax1.set_xlim([min(name_z.keys()) - 1, max(name_z.keys()) + 1])
        ax2.set_ylim([-0.72, 0.72])
        ax1.tick_params(which="both", right=True, labelsize=FONTSIZE)
        ax1.minorticks_on()
        ax2.tick_params(which="both", right=True, labelsize=FONTSIZE)
        ax2.minorticks_on()

        # Change x-axis. Use only odd numbers. 6, 10, ... for major
        # 8, 12... for minor
        x_axis_maj = [z for z in name_z.keys() if (z - 2) % 4 == 0]
        x_axis_maj_labs = [name_z[z] for z in name_z.keys() if (z - 2) % 4 == 0]
        x_axis_min = [z for z in name_z.keys() if z % 4 == 0]
        x_axis_min_labs = [name_z[z] for z in name_z.keys() if z % 4 == 0]

        # Major ticks
        ax2.set_xticks(x_axis_maj)
        ax2.set_xticklabels(x_axis_maj_labs, size=FONTSIZE)

        # Minor ticks
        ax2.set_xticks(x_axis_min, minor=True)
        ax2.set_xticklabels(x_axis_min_labs, minor=True, size=FONTSIZE)

        # Displace the minor ticks down
        ax2.xaxis.set_tick_params(which="minor", pad=20)

        # Aling y-labels
        fig.align_ylabels()

        # Put legend outside of plot
        ncol = min(6, n_plots + 1)
        print(f"Plotting: {key}")
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.75), ncol=ncol,
                    fontsize=FONTSIZE)

        # Create plotting directory if it does not exist
        if not os.path.isdir(pathn):
            os.mkdir(pathn)

        # Save the plot
        filename = os.path.join(pathn, key + ".pdf")
        plt.savefig(filename)
        filename = os.path.join(pathn, key + ".png")
        plt.savefig(filename)

        # Close the figure to save memory
        plt.close()

def main():
    """
    Program for plotting the outputs from the classification
    """

    if len(sys.argv) < 3:
        s = "Incorrect number of arguments. "
        s += f"Use: python3 {sys.argv[0]} <file1> [file2 ...] <directory>"
        sys.exit(s)

    # Save files with data and directory
    files = sys.argv[1:-1]
    pathn = sys.argv[-1]

    # Define all the directories
    dir_data = "Ba_star_classification_data"
    fruity_mods = "models_fruity"
    monash_mods = "models_monash"
    data_file = "all_abund_and_masses.dat"

    fruity_dir = os.path.join(dir_data, fruity_mods)
    monash_dir = os.path.join(dir_data, monash_mods)
    data_file = os.path.join(dir_data, data_file)

    # Sort files in fruity and monash
    predicted_models_dict = get_dict_predicted(files)

    # Load the stellar abundances
    dict_data = get_data_values(data_file)

    # The fruity and monash models
    fruity_models_dict = get_data_fruity(fruity_dir)
    monash_models_dict = get_data_monash(monash_dir)

    # Load the red elements
    with open("element_set.dat", "r") as fread:
        for line in fread:
            lnlst = line.split()

            # Skip comments and empty lines
            if len(lnlst) == 0 or "#" in lnlst[0]:
                continue

            red_elements = lnlst
            break

    # Remove "/Fe"
    red_elements = [elem.split("/")[0] for elem in red_elements]
    red_elements.remove("Fe")

    # Now that all the data is loaded, we can plot it
    plot_results(predicted_models_dict, fruity_models_dict,
                 monash_models_dict, dict_data, red_elements, pathn)

if __name__ == "__main__":
    main()
