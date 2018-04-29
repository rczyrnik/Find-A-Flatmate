# BASICS
from time import time
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# HELPER FUNCTIONS
import my_resample as ms
import my_functions as mf

# METRICS
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# CLASSIFIERS
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# GRID SEARCHING
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# READ DATA
try:
    data_file_path = "/Users/gandalf/Documents/coding/do_not_commit/capstone/"
    website_file_path = '/Users/gandalf/Documents/coding/rczyrnik.github.io/capstone/'
    X_train = np.load(data_file_path+'X_train.npy')
    X_test = np.load(data_file_path+'X_test.npy')
    y_train = np.load(data_file_path+'y_train.npy')
    y_test = np.load(data_file_path+'y_test.npy')
    cols = np.load(data_file_path+'cols.npy')
except:
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    cols = np.load('cols.npy')

# # RESAMPLE
# X_train, y_train = ms.oversample(X_train, y_train, .5)

# GRID VARIABLES

random_forest_grid={
    # "n_estimators": [11],
    # "max_features": [.6, .7, .8, .9],
    # "max_depth": [14, 16, 18],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    # "max_leaf_nodes": [450, 500, 550, 600],
    # "min_impurity_decrease": [0, .00005, .0001, .00015],
    # "class_weight": [{0:.9, 1:.1}, {0:.7, 1:.3}, {0:.5, 1:.5}],
}

# GRIDSEARCH
rfc = RandomForestClassifier()
grid_search = GridSearchCV(rfc, param_grid=random_forest_grid, scoring='f1', n_jobs=-1, cv=10, return_train_score=False)
grid_search.fit(X_train, y_train)

# PRINT RESULT
print(grid_search.best_params_)

# SAVE AS DF
df = pd.DataFrame(grid_search.cv_results_)
# df = df.drop(['params'], axis=1)
df.to_pickle('gs_random_forest.pkl')
