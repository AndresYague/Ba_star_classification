import numpy as np, sys, os, pandas as pd
import itertools as iter
from sklearn.tree import export_graphviz
import pydot
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd


def models_file(mod_dir, nondil=False):
    '''Search for the proper processed models file'''
    data_directory = os.path.join(sys.path[0], "data_processing_and_plotting")
    if "fruity" in mod_dir:
        if nondil == False:
            models_file = "processed_models_fruity.txt"
        else: models_file = "processed_nondil_models_fruity.txt"
        models_file = os.path.join(data_directory, models_file)
    elif "monash" in mod_dir:
        if nondil == False:
            models_file = "processed_models_monash.txt"
        else: models_file = "processed_nondil_models_monash.txt"
        models_file = os.path.join(data_directory, models_file)

    return models_file

def df_normalize(df_lst, exclude_col_lst):
    '''Normalizing a list of df-s *with the same factors* of the first df!'''
    df_norm_lst = []
    for kk in range(len(df_lst)):
        df = df_lst[kk]
        exclude_col = exclude_col_lst [kk]

        if kk == 0: # calculating the mean and std only for the first df
            mean = df.mean()
            std = df.std()

        if exclude_col != '': # the label names are not to be normalized
            donottouch = df.pop(exclude_col)

        if 'err' in df.columns.values[0]:  # if the df of the errors, an '_err' suffix has to be added to the column names
            mean_err = pd.Series(mean.values, index=[x+'_err' for x in mean.keys()])
            std_err  = pd.Series(std.values,  index=[x+'_err' for x in std.keys()])
            df = (df-mean_err)/std_err
        else:
            df = (df - mean) / std

        if exclude_col != '':
            df[exclude_col] = donottouch
        df_norm_lst.append(df)
    return df_norm_lst

def df_normalize2(df_lst, exclude_col_lst, norm='n'):
    '''Normalizing a list of df-s *with the same factors* of the first df!
    norm: n -> normalization (min, max); s -> standardization (avg, std)'''
    df_norm_lst = []
    for kk in range(len(df_lst)):
        df = df_lst[kk]
        exclude_col = exclude_col_lst [kk]

        if exclude_col != '': # the label names are not to be normalized
            donottouch = df.pop(exclude_col)

        if kk == 0: # calculating the mean and std only for the first df
            mean = df.mean()
            std = df.std()
            minn = df.min()
            maxx = df.max()


        if 'err' in df.columns.values[0]:  # if the df of the errors, an '_err' suffix has to be added to the column names
            mean_err = pd.Series(mean.values, index=[x+'_err' for x in mean.keys()])
            std_err  = pd.Series(std.values,  index=[x+'_err' for x in std.keys()])
            min_err = pd.Series(minn.values, index=[x + '_err' for x in minn.keys()])
            max_err = pd.Series(maxx.values, index=[x + '_err' for x in maxx.keys()])
            df = (df-mean_err)/std_err
        else:
            if norm=='s': df = (df - mean) / std
            elif norm=='n': df = (df - minn) / (maxx-minn)

        if exclude_col != '':
            df[exclude_col] = donottouch
        df_norm_lst.append(df)
    return df_norm_lst

def df_reader(location):
    ''' Reads in the processed models to a DataFrame '''
    with open(location) as myfile: headRow = next(myfile)
    columns = [x.strip() for x in headRow.split(' ')]

    df = pd.read_csv(location, delim_whitespace=True, names=columns[1:-1], skiprows=1,
                     na_values=-9999.0)
    return df


def df_reader_obs(location, meanfill=True):
    ''' Reads in the observations and errors to a DataFrame '''
    with open(location) as myfile: headRow = next(myfile)
    columns = [x.strip() for x in headRow.split(' ')]

    df = pd.read_csv(location, delim_whitespace=True, names=columns[1:], skiprows=1, na_values='-')
    df_err = df.filter(regex=r'_err$')
    err_headers = list(df_err.columns.values)
    err_headers.append('Name')
    starnames = list(df.filter(like='Name').Name)
    df.drop(columns=err_headers, axis=1, inplace=True)

    if meanfill == True: df.fillna(df.mean(), inplace=True) # replace the missing values with mean
    df_err.fillna(1, inplace=True)

    return df, df_err, starnames

def feature_subtract(df, order=1):
    '''Subtracts all number-columns in all combinations'''
    df_new = df.copy()
    newcols = []
    for a, b in iter.combinations(df.columns, 2):
        if (a and b) != 'Label' and (a or b) not in ['Fe/H', 'Fe/H_err']:   # labels are not to be subtracted :)
            if a[:1] == 'Y':    newcol = '{}/{}'.format(a[:1], b[:2]) # Y is only one character...
            elif b[:1] == 'Y':  newcol = '{}/{}'.format(a[:2], b[:1])
            else:               newcol = '{}/{}'.format(a[:2], b[:2])
            df_new[newcol] = df[a] - df[b]
            newcols.append(newcol)
    if order == 2:
        for a, b in iter.combinations(newcols, 2):
            newcol = '{}-{}'.format(a, b)
            if a[:2] != b[:2]: # if the first two elements are the same, then [a/b]-[a/c] = [c/b]
                df_new[newcol] = df_new[a] - df_new[b]
    return df_new

def rfclassifier(df, labels):
    classifier = RandomForestClassifier(n_estimators=800, max_depth=8, min_samples_split=2,
                                      min_samples_leaf=2, max_features='auto', bootstrap=True, n_jobs=-1) # max_depth:18
    print("Now fitting the classifier")
    classifier.fit(df, labels)

    return classifier

def importances(df, classifier):
    importances = classifier.feature_importances_
    importances = pd.Series(importances, index=df.columns)
    return importances

def gbclassifier(df, labels):
    gbm = GradientBoostingClassifier(verbose=2, learning_rate=0.06, n_estimators=20, tol=1e-5,
                                     max_depth=10, min_samples_split=3) # n_estimators=30, max_depth=11, min_samples_split=3,
                                     #tol=1e-5, max_features=5, min_samples_leaf=26, subsample=0.78
    gbm.fit(df, labels)
    return gbm

def xgbclassifier(df_tr, labels, label_encoder, num_labs):
    print('Now XGB')
    #label_encoder = LabelEncoder()
    #label_encoder = label_encoder.fit(labels)
    lab_tr = label_encoder.transform(labels)

    #df_tr, df_tst, lab_tr, lab_tst = train_test_split(df, labels_encoded,
     #                                                 test_size=0.2)  # Training and test sets
    #dmatrix = xgb.DMatrix(data=df, label=labels)
    # xgb_mod = XGBClassifier(objective = 'binary:logistic', colsample_bytree = 0.5, learning_rate = 0.2, max_depth = 5,
    #                        alpha = 10, n_estimators = 10, use_label_encoder=False, verbosity=0)
    # xgb_mod = XGBClassifier(learning_rate = 0.05, gamma = 0, colsample_bytree = 0.9, max_depth = 13,
    #                         n_estimators = 13, subsample = 0.9, objective = 'multi_softmax',
    #                         random_state=42, use_label_encoder=False, verbosity=2) # alpha = 0.08,
    xgb_mod = XGBClassifier(objective='multi_softmax', learning_rate=0.05, use_label_encoder=False, n_jobs=-1,
                            max_depth=5, gamma=2, n_estimators=80, reg_lambda=2) # l-rate=0.02, n_estim = 100
    xgb_mod.fit(df_tr, lab_tr, eval_metric='rmse')
    # lab_pred = xgb_mod.predict(df_tst)
    # count = 0
    # right = 0
    # for kk in range(len(lab_pred)):
    #     if lab_pred[kk] == lab_tst[kk]: right += 1
    #     count +=1
    # print(right/count)
    # lab_pred = label_encoder.inverse_transform(lab_pred)

    return xgb_mod

def xgb_pred(label_encoder, xgb_mod, lab_tst, df_tst):
    labels_encoded = label_encoder.transform(lab_tst)
    lab_pred = xgb_mod.predict(df_tst)







def visualise_tree(ii, df_tr, classifier):
    if ii==0:
        feature_list = list(df_tr.columns)
        # Pull out one tree from the forest
        tree = classifier.estimators_[42]
        # Export the image to a dot file
        export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=2,
                        filled=True, class_names = classifier.classes_)
        # Use dot file to create a graph
        (graph,) = pydot.graph_from_dot_file('tree.dot')
        # Write graph to a png file
        graph.write_png('tree.png')


class Logger(object):  # logging to console and output file at once
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.terminal = sys.stdout
        self.log = open(file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self): pass
