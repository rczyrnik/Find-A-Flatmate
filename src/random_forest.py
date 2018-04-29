# IMPORTS
import pandas as pd
import numpy as np

import my_resample as ms
import my_functions as mf

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier

import pickle

data_file_path = "/Users/gandalf/Documents/coding/do_not_commit/capstone/"
website_file_path = '/Users/gandalf/Documents/coding/rczyrnik.github.io/capstone/'
# data_file_path = ""

X_train = np.load(data_file_path+'X_train.npy')
X_test = np.load(data_file_path+'X_test.npy')
y_train = np.load(data_file_path+'y_train.npy')
y_test = np.load(data_file_path+'y_test.npy')
cols = np.load(data_file_path+'cols.npy')

def my_random_forest(X_train, y_train, resamp):
    X_train, y_train = ms.oversample(X_train, y_train, resamp)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return recall, precision, f1

def gridsearch_gbc(X_train, y_train, resamp,
                        resample_wt=.5,
                        learning_rate = .1,
                        max_depth=3,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        subsample=1,
                        max_features=None,
                        max_leaf_nodes=None,
                        min_impurity_decrease=0,
                        loss='deviance',
                        criterion='friedman_mse',
                        ):

    X_train, y_train = ms.oversample(X_train, y_train, resamp)

    model = GradientBoostingClassifier(learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        subsample=subsample,
                                        max_features=max_features,
                                        max_leaf_nodes=max_leaf_nodes,
                                        min_impurity_decrease=min_impurity_decrease,
                                        loss=loss,
                                        criterion=criterion,
                                        )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return model.get_params(), recall, precision, f1

def my_gridsearch(iterations=10):
    my_dic = {}
    for resamp in np.arange(.1, 1, .1):
        recalls, precisions, f1s = [],[],[]
        for n in range(iterations):
            parameters, recall, precision, f1 = gridsearch_gbc(X_train, y_train, resamp)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
        print(parameters)
        # my_dic[(parameters, resamp)] = mf.get_stats([recalls, precisions, f1s])
    return my_dic

if __name__ == "__main__":
    results = my_gridsearch()
    mf.save_obj(results, "first_results")
