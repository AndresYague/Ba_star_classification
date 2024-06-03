from plot_lib import *
from processplot_data_lib import *
from plot_correl_funcs import *
import os

# INPUTS ----------------------------------------------------------------
x_axis = 'feh'      # Quantity on x-axis: res/obs/feh/mass/dil
y_axis = 'obs'      # Quantity on y-axis: res/obs
df_resids = pd.read_csv('boxplot_resids.dat') # input file containing the residuals
SAVE_DIR = "correl_forgit" # name of output directory
# DEFINE BELOW (LINES 94 ONWARDS) WHICH COMBINATIONS OF PLOTS YOU WANT FOR YOUR SPECIFIC COMBINATION OF X-Y VALUE
# (uncomment if anything is unnecessary)

only_NbZr = False    # if only Nb vs Zr plot
onlyBa = False       # if only Nb vs Zr: True if only Ba star points, False if AGB too
Nb_lim = -1          # negative if not Zr-thermometer, 0 if omega-residual, positive if thermometer

cmass = False       # if colour the points according to the mass of the stars
solZrNb = 1.18      # solar Zr/Nb

config = {"solZrNb": solZrNb, "SAVE_DIR": SAVE_DIR, "onlyBa": onlyBa, "Nb_lim": Nb_lim} # this will be loaded in the functions module
set_config(config)

# axis limits for different quantities
if x_axis == "feh": xmin, xmax = -0.8, 0.3
elif x_axis == "obs": xmin, xmax = -0.5, 1.5
elif x_axis == "res": xmin, xmax = -0.8, 0.7
elif x_axis == "mass": xmin, xmax = 1, 4
elif x_axis == "dil": xmin, xmax = 0, 1
if y_axis == "obs": ymin, ymax = -0.5, 1.5
elif y_axis == "res": ymin, ymax = -0.8, 0.7
lims = [xmin, xmax, ymin, ymax]

font = {'family': 'sans-serif', 'size': 34}
matplotlib.rc('font', **font)


peak1 = ['Rb', 'Sr', 'Y' ,'Zr', 'Nb', 'Mo', 'Ru']
#peak1 = ['Sr', 'Y' ,'Zr', 'Nb', 'Mo']
# peak1 = [val for val in peak1 if val in red_elements]
peak2 = ['La', 'Ce', 'Nd', 'Sm', 'Eu']
if only_NbZr:
    peak1 = ['Nb']
    peak2 = ['Zr']

# write out the variables defined in the main plotting file but used in the correl_funcs
#tmp_file_variables.writelines([str(solZrNb)+"\n", save_dir])

solZrNb = 1.18
save_dir = "correl_forgit"


# LOAD DATA --------------------------------------------------------------
# Define all the directories
cwd = os.getcwd()
dir_data = os.path.join(cwd,"../Ba_star_classification_data")
fruity_mods = "models_fruity"
monash_mods = "models_monash"
data_file = "all_data_w_err.dat"

fruity_dir = os.path.join(dir_data, fruity_mods)
monash_dir = os.path.join(dir_data, monash_mods)
data_file = os.path.join(dir_data, data_file)

# Load the stellar abundances
dict_data = get_data_values(data_file)
df_obs = conv_dict_to_df(dict_data)
obs_elems = df_obs.columns.values.tolist()
df_obs = feature_subtract(df_obs[[i for i in obs_elems if "err" not in i]], 1).T
df_obs_Fe = copy.deepcopy(df_obs.T)[[i for i in df_obs.T.columns if "/Fe" in i]].T
stars_lst = df_obs.index.tolist()
df_obs2 = pd.DataFrame.reset_index(df_obs.T).rename(columns={'index': 'star'})
df_obs2['star'] = df_obs2['star'].astype(str)
df_resids['star'] = df_resids['star'].astype(str)
df_resid_obs = pd.DataFrame.merge(df_resids, df_obs2, on='star', how='outer', suffixes=('_res', '_obs')).T
df_resid_Fe = df_resid_obs.T[[i for i in df_resid_obs.T.columns if "/Fe_res" in i]].T

peak1p1_res, peak1p2_res, peak2p1_res, peak2p2_res, peak2p2_res_reorder = peakfilter(df_resid_obs, peak1, peak2, res_obs_mode=True)
for df in [peak1p1_res, peak1p2_res, peak2p1_res, peak2p2_res, df_resid_Fe]:
    df.index = df.index.str.replace(r'_res', '')

peak1p1_obs, peak1p2_obs, peak2p1_obs, peak2p2_obs, peak2p2_obs_reorder = peakfilter(df_obs, peak1, peak2)
peak1Fe_obs, peak2Fe_obs = separate_Fe_peaks(df_obs_Fe, peak1, peak2)
peak1Fe_all, peak2Fe_all = separate_Fe_peaks(df_resid_obs, peak1, peak2)
peak1Fe_all.loc['dil'] = df_resid_obs.loc['dil']

df_obs = df_obs[[i for i in df_obs.columns if "/Fe" not in i]]
df_obs_plot = df_obs.T
if y_axis == 'res':
    FeH = df_resid_obs.T['Fe/H_obs']
else:
    FeH = df_obs.T['Fe/H']
    CeFe = df_obs.T['Ce/Fe']

# calculating mean abundances/dilutions for each star from all classified models, for Nb-Zr plots
peak1Fe_mean = copy.deepcopy(peak1Fe_all).T
peak2Fe_mean = copy.deepcopy(peak2Fe_all).T
peak1Fe_mean = create_mean(peak1Fe_mean, df_resid_obs)
peak2Fe_mean = create_mean(peak2Fe_mean, df_resid_obs)

mass = df_resids['mass']
unique(peak1Fe_mean.loc['star'])
a = peak1Fe_mean.T[peak1Fe_mean.T['Nb/Fe_res'].isnull()]


# CASES OF WHAT TO PLOT FOR EACH X-Y COMBINATION -------------------------------------------------
if x_axis == "feh" and y_axis == "obs":
    # # Abundances vs FeH
    # plot_peak(FeH, df_obs_Fe, "feh", "obs", "Fe", 2, len(peak1), mass, p2p1=True, absFe=True)
    # # Peak 1 / Peak 1
    # plot_peak(FeH, peak1p1_obs, "feh", "obs", "p1p1", len(peak1), len(peak1), mass)
    # # Peak 2 / Peak 1
    # plot_peak(FeH, peak2p1_obs, "feh", "obs", "p2p1", len(peak2), len(peak1), mass, p2p1=True)
    # Peak 1 / Peak 2
    plot_peak(FeH, peak1p2_obs, "feh", "obs", "p1p2", len(peak1), len(peak2), mass, p2p1=True)
    # # Peak 2 / Peak 2
    # plot_peak(FeH, peak2p2_obs, "feh", "obs", "p2p2", len(peak2), len(peak2), mass)
    # # Peak 2 / Peak 2: reversed for placing below Peak 1 / Peak 1 figure, if wanted to (that placing is only with external image editor, sorry)
    # plot_peak(FeH, peak2p2_obs_reorder, "feh", "obs", "p2p2_flip", len(peak2), len(peak2), lims, mass, triag2=True)


if x_axis == "feh" and y_axis == "res":
    # Abundance residuals vs FeH
    plot_peak(FeH, df_resid_Fe, "feh", "res", "Fe", 2, len(peak1), mass, p2p1=True, absFe=True)
    # Peak 1 / Peak 1 residuals
    plot_peak(FeH, peak1p1_res, "feh", "res", "p1p1", len(peak1), len(peak1), mass)
    # Peak 2 / Peak 1 residuals
    plot_peak(FeH, peak2p1_res, "feh", "res", "p2p1", len(peak2), len(peak1), mass, p2p1=True)
    # Peak 1 / Peak 2 residuals
    plot_peak(FeH, peak1p2_res, "feh", "res", "p1p2", len(peak1), len(peak2), mass, p2p1=True)
    # Peak 2 / Peak 2 residuals
    plot_peak(FeH, peak2p2_res, "feh", "res", "p2p2", len(peak2), len(peak2), mass)
    # Peak 2 / Peak 2 residuals, reversed triangle
    plot_peak(FeH, peak2p2_res_reorder, "feh", "res", "p2p2_flip", len(peak2), len(peak2), mass, triag2=True)


if x_axis == "obs" and y_axis == "obs":
    # Abundances vs Ce/Fe (for main text)
    plot_peak(CeFe, df_obs_Fe, "feh", "obs", "CeFe", 2, len(peak1), mass, p2p1=True, absFe=True)# Peak 1 vs Peak 1 abundances
    # Peak 1 vs Peak 1 abundances
    plot_peak(peak1Fe_obs, peak1Fe_obs, "obs", "obs", "p1_p1", len(peak1), len(peak1), mass)
    # Peak 2 vs Peak 1 abundances
    plot_peak(peak1Fe_obs, peak2Fe_obs,"obs", "obs", "p2_p1", len(peak2), len(peak1), mass, p2p1=True)
    # Peak 1 vs Peak 2 abundances
    plot_peak(peak2Fe_obs, peak1Fe_obs, "obs", "obs", "p1_p2", len(peak1), len(peak2), mass, p2p1=True)
    # Peak 2 vs Peak 2 abundances
    plot_peak(peak2Fe_obs, peak2Fe_obs, "obs", "obs", "p2_p2", len(peak2), len(peak2), mass)
    # Peak 2 vs Peak 2 abundances, reversed triangle
    plot_peak(peak2Fe_obs, peak2Fe_obs, "obs", "obs", "p2_p2_flip", len(peak2), len(peak2), mass, triag2=True)

    # Only Nb vs Zr
    #plot_peak(peak1Fe_mean, peak2Fe_mean, "obs", "obs", "NbZr_Ba_AGB", len(peak2), len(peak1), mass, p2p1=True)
    plot_peak(peak1Fe_mean, peak2Fe_mean, "obs", "obs", "NbZr_om_res", len(peak2), len(peak1), mass, p2p1=True)



if x_axis == "obs" and y_axis == "res":
    # Peak 1 residual vs Peak 1 obs
    plot_peak(peak1Fe_all, peak1Fe_all, "obs", "res", "p1_p1", len(peak1), len(peak1), mass, p2p1=True)
    # Peak 2 residual vs Peak 1 obs
    plot_peak(peak1Fe_all, peak2Fe_all, "obs", "res", "p2_p1", len(peak2), len(peak1), mass, p2p1=True)
    # Peak 1 residual vs Peak 2 obs
    plot_peak(peak2Fe_all, peak1Fe_all, "obs", "res", "p1_p2", len(peak1), len(peak2), mass, p2p1=True)
    # Peak 2 residual vs Peak 2 obs
    plot_peak(peak2Fe_all, peak2Fe_all, "obs", "res", "p2_p2", len(peak2), len(peak2), mass, p2p1=True)
    # Nb and Mo residual vs Peak 1 obs (for main text)
    plot_peak(peak1Fe_all, peak1Fe_all[peak1Fe_all.index.str.contains('Nb|Mo')], "obs", "res", "p1_p1_Nb_Mo", 2, len(peak1), mass, p2p1=True)


if x_axis == "res" and y_axis == "res":
    # Peak 1 vs Peak 1 residuals -- only 5 elements for main text: please set peak 1 at the beginning of this file and the functions module
    plot_peak(peak1Fe_all, peak1Fe_all, "res", "res", "p1p1-only5elems", len(peak1), len(peak1), mass)
    # Peak 1 vs Peak 1 residuals
    plot_peak(peak1Fe_all, peak1Fe_all, "res", "res", "p1p1", len(peak1), len(peak1), mass)
    # Peak 1 vs Peak 2 residuals
    plot_peak(peak1Fe_all, peak2Fe_all, "res", "res", "p2p1", len(peak2), len(peak1), mass, p2p1=True)
    # Peak 2 vs Peak 1 residuals
    plot_peak(peak2Fe_all, peak1Fe_all, "res", "res", "p1p2", len(peak1), len(peak2), mass, p2p1=True)
    # Peak 2 vs Peak 2 residuals
    plot_peak(peak2Fe_all, peak2Fe_all, "res", "res", "p2p2", len(peak2), len(peak2), mass)
    # Peak 2 vs Peak 2 residuals, reversed triangle
    plot_peak(peak2Fe_all, peak2Fe_all, "res", "res", "p2p2_flip", len(peak2), len(peak2), mass, triag2=True)

if x_axis == "mass" and y_axis == "res":
    plot_peak(df_resid_obs, df_resid_Fe, "mass", "res", "massresp1", 2, len(peak1), mass, p2p1=True, absFe=True)

if x_axis == "dil":
    if y_axis == "res":
        plot_peak(df_resid_obs, df_resid_Fe, "dil", "res", "dilresp1", 2, len(peak1), mass, p2p1=True, absFe=True)
    if y_axis == "obs":
        plot_peak(df_resid_obs, df_resid_obs, "dil", "obs", "dilobs", 2, len(peak1), mass, p2p1=True, absFe=True)
