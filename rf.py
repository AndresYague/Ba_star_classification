import numpy as np, sys, pandas as pd
import rf_lib as rfl
from classify_lib import *
import joblib
import matplotlib
from matplotlib import pyplot as plt

from sklearn import model_selection, metrics
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

order = 1   # subtraction order, if 1, [A/B] is used, if 2, [A/B]-[C/D] too
ntry = 5    # number of RF classifiers that are used for the evaluation
fit_if_min_classifier = 3   # if the probability is above probability limit in fit_if_min_classifier number classifiers, counts as a potential polluter AGB
probability_limit = 0.075
normalise = False # True if features are normalised, for RF, unnecessary
visualize = False # True if create image of a decision tree
mode = "rf"       # "rf" or "gb" for random forest or gradient boosting; latter is under construction
mod_dir = "rf_forgit_fruity" # output directory; please include the term "fruity" or "monash" accordingly

font = {'family': 'sans-serif', 'size': 21}
matplotlib.rc('font', **font)

# Create or load model
importance_figname = 'importance-{:}'.format(mod_dir)   # figure filename for feature importances

if not os.path.exists(mod_dir): os.makedirs(mod_dir)    # create a directory, if not existing
file_path = os.path.join(mod_dir, '{:}-out.txt'.format(mod_dir))

sys.stdout = rfl.Logger(file_path) # logging to both console and file

np.random.seed()


# READ IN THE MODELS ------------------------------------
if ("fruity" or "monash") not in mod_dir:
    exit("Please include the name fruity or monash in the directory name")

models_file = rfl.models_file(mod_dir)  # Select the file of the processed models
df_ori = rfl.df_reader(models_file)     # Original DataFrame containing all the model abundances and labels

nondil_models_file = rfl.models_file(mod_dir, nondil=True) # Non-diluted models made with the preprocess, for GoF
df_nondil = rfl.df_reader(nondil_models_file)
df_nondil = df_nondil[df_ori.columns.values]
labels_nondil = df_nondil['Label']
num_labs = len(labels_nondil)

if mode == "xgb": meanfill = False
else: meanfill = True
df_obs, df_err, starnames = rfl.df_reader_obs("data_processing_and_plotting/processed_data.txt", meanfill) # Df for observations and errors
df_obs_nonnorm, df_err_nonnorm, starnames = rfl.df_reader_obs("data_processing_and_plotting/processed_data.txt", meanfill)

df_models_extraf = rfl.feature_subtract(df_ori, order=order)   # DataFrame extended with the modified inputs, extra features
df_obs_extraf = rfl.feature_subtract(df_obs, order=order)
df_err_extraf = rfl.feature_subtract(df_err, order=order)

if normalise:
    [df_models_extraf, df_obs_extraf, df_err_extraf] = rfl.df_normalize([df_models_extraf, df_obs_extraf, df_err_extraf], ['Label', '', ''])


labels = df_models_extraf.pop('Label')        # Separate the labels to a list


# FIT THE MODEL ------------------------------------------------------------------
dirname = mod_dir
lab_enc = LabelEncoder() # label encoding for XGB
lab_enc = lab_enc.fit(labels)


for ii in range(ntry): # for ntry numbers of different classifiers with separate randomisation
    df_tr, df_tst, lab_tr, lab_tst = train_test_split(df_models_extraf, labels, test_size=0.2)  # Training and test sets

    dirname_curr = os.path.join(dirname, "{:}_{:}.joblib".format(mode, ii)) # current path for the model to be saved
    if os.path.exists(dirname_curr) == False: # if the classifier was already created
        if mode == "rf": classifier = rfl.rfclassifier(df_tr, lab_tr)
        elif mode == "gb": classifier = rfl.gbclassifier(df_tr, lab_tr)
        elif mode == "xgb": classifier = rfl.xgbclassifier(df_tr, lab_tr, lab_enc, num_labs)
        joblib.dump(classifier, dirname_curr)    # save the classifier
    else: classifier = joblib.load(dirname_curr) # if it exists, just load it

    if visualize: rfl.visualise_tree(ii, df_tr, classifier) # if a decision tree is to be visualised

    lab_tst_pred = classifier.predict(df_tst)  # predictions to test the classifier
    if mode == "xgb": lab_tst_pred = lab_enc.inverse_transform(lab_tst_pred) # XGB gives back predictions in encoded format
    print('Accuracy: ', metrics.accuracy_score(lab_tst, lab_tst_pred))
    print()

    lab_stars_pred_now = pd.DataFrame(classifier.predict_proba(df_obs_extraf)) # predictions for current classifier
    importance_now = rfl.importances(df_tr, classifier) # feature importances for current classifier

    lab_stars_count_now = lab_stars_pred_now.copy() # in this list, each item that is considered as a good fit, will be 1, the others 0
    lab_stars_count_now[lab_stars_count_now >= probability_limit] = 1 # plus one model is accurate if proba is > probability_limit
    lab_stars_count_now[lab_stars_count_now < probability_limit] = 0

    if ii == 0:  # if first classifier, initialize labels and importances
        lab_stars_pred = lab_stars_pred_now.div(ntry)
        lab_stars_count = lab_stars_count_now.copy()
        importances = importance_now.div(ntry) # importances are averaged over the n random models
    else:        # all predictions count to the average, should be divided by ntry
        lab_stars_pred = lab_stars_pred.add(lab_stars_pred_now.div(ntry))
        lab_stars_count = lab_stars_count.add(lab_stars_count_now)
        importances = importances.add(importance_now.div(ntry))


    # IMPORTANCES PLOT ----------------------------------------------
    if order == 1:
        fig, ax = plt.subplots(figsize=(20, 10))
    elif order == 2: # make the plot wider
        fig, ax = plt.subplots(figsize=(50, 10))

    importances.plot.bar(ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(dirname, importance_figname))


# PREDICT FOR THE STARS --------------------------------------------
for labs_df in [lab_stars_count, lab_stars_pred]:
    if mode == "xgb": labs_df.columns = lab_enc.inverse_transform(classifier.classes_)
    else: labs_df.columns = classifier.classes_ # The column names are the labels of the models

for ii in range(len(starnames)): # for each star
    # Initialize star instance for current star ----------------
    curr_name = starnames[ii] # name of current star
    curr_data = list(df_obs_nonnorm.iloc[ii])
    curr_err  = list(df_err_nonnorm.iloc[ii])
    star_instance = StarStat(curr_name, curr_data, curr_err)

    # Select the predictions to print for the current star ------------
    curr_preds = lab_stars_pred.iloc[ii].sort_values(ascending=False) # Sort the predictions by probability for the star
    curr_counts = lab_stars_count.iloc[ii].sort_values(ascending=False)
    curr_model_dic = curr_preds[curr_counts >= fit_if_min_classifier]
    predcount = len(curr_model_dic)-1
    curr_model_names = curr_model_dic.keys()
    print("For star {}:".format(starnames[ii]))

    # For each probable model, calculate GoF and dil ----------------
    for modelnum in range(len(curr_model_dic)):
        curr_label = curr_model_names[modelnum]
        curr_model = np.asfarray((df_nondil.iloc[labels_nondil[labels_nondil == curr_label].index[0]])[:-1])
                # Read in the current probable model from the non-diluted set

        pVal, dilution = star_instance.calculate_dilution(curr_model, max_dil=0.9) # Calculate GoF and dil
        if pVal > 0.1 and dilution < 0.89: # dilution < 0.89 and
            s = f"Label {curr_label} with goodness of fit {pVal * 100:.2f}%"
            s += f" and dilution {dilution:.2f}"
            s += f" , probability {curr_preds.loc[curr_label]:.2f}"
            print(s)

