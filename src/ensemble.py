'''
ensemble the models
'''


# ----------------------------------- IMPORTS ----------------------------------
# BASICS

import datetime
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# HELPER FUNCTIONS
import my_resample as ms
import my_functions as mf

# METRICS
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# CLASSIFIERS
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

# GRID SEARCHING
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# WARNINGS
import warnings
warnings.filterwarnings('ignore')


# ----------------------------------- SETUP ----------------------------------

# read the data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
cols = np.load('cols.npy')

# create a dictionary to hold model info
ed = {}


# ---------------------------- Gradient Boost ----------------------------------


print("\n")
print(datetime.datetime.now())
print("Starting Gradient Boosting")

# Grid
gradient_boost_grid = {
    "learning_rate": np.arange(.1, 2, .1),
    "n_estimators": np.arange(1, 100, 10),
    "max_depth": range(1, 10),                   #  3
    "min_samples_split": [2, 6, 20, 60],      # 2
    "min_samples_leaf": [1, 5, 10, 50],       # 1
    "max_features": ["sqrt", "log2", None], # None
    "max_leaf_nodes": [10, 100],              # None
    "min_impurity_decrease": [.1, .3, .5, .7, .9],    # 0
}

# Grid Search
gbc = GradientBoostingClassifier()
gb_model = RandomizedSearchCV(gbc, param_distributions=gradient_boost_grid, scoring='f1')
gb_model.fit(X_train, y_train)

# Predict
y_pred_gb = gb_model.predict(X_test)
y_pp_gb = gb_model.predict_proba(X_test)

print("\nGRADIENT BOOSTING METRICS")
print("Model recall: {:.3f}".format(recall_score(y_test, y_pred_gb)))
print("Model precision: {:.3f}".format(precision_score(y_test, y_pred_gb)))
print("Model f1: {:.3f}".format(f1_score(y_test, y_pred_gb)))
print("Model accuracy: {:.3f}".format(gb_model.score(X_test, y_test)))

print ("\nCONFUSION MATRIX")
print (confusion_matrix(y_test, y_pred_gb))
print ("\nkey:")
print (" TN   FP ")
print (" FN   TP ")

# ---------------------------- ADA BOOST ----------------------------------
print("\n")
print(datetime.datetime.now())
print("Starting Ada Boosting")

# Grid
ada_boost_grid = {
    "n_estimators": range(1, 150, 10),          #  50
    "learning_rate": np.arange(.1, 2.1, .2),      #  1
}

# Grid Search
abc = AdaBoostClassifier()
ab_model = RandomizedSearchCV(abc, param_distributions=ada_boost_grid, scoring='f1', n_jobs=-1)
ab_model.fit(X_train, y_train)

# Predict
y_pred_ab = ab_model.predict(X_test)
y_pp_ab = ab_model.predict_proba(X_test)

print("\nADA BOOST METRICS")
print("Model recall: {:.3f}".format(recall_score(y_test, y_pred_ab)))
print("Model precision: {:.3f}".format(precision_score(y_test, y_pred_ab)))
print("Model f1: {:.3f}".format(f1_score(y_test, y_pred_ab)))
print("Model accuracy: {:.3f}".format(ab_model.score(X_test, y_test)))

print ("\nCONFUSION MATRIX")
print (confusion_matrix(y_test, y_pred_ab))
print ("\nkey:")
print (" TN   FP ")
print (" FN   TP ")

# ---------------------------- RANDOM FOREST ----------------------------------

print("\n")
print(datetime.datetime.now())
print("Starting Random Forest")

# Grid
random_forest_grid={
    "n_estimators": [5, 7, 11],
    "max_features": [.6, .7, .8, .9],
    "max_depth": np.arange(11, 17, 1),
    "min_samples_split": np.arange(2, 20),
    "min_samples_leaf": (1, 5),
    "max_leaf_nodes": np.arange(200, 600, 10),
    "min_impurity_decrease": np.arange(0, .0001, .00001),
    "class_weight": [{0:n, 1:1-n} for n in np.arange(.1, 1, .1)]
}

# Grid Search
rfc = RandomForestClassifier()
rf_model = RandomizedSearchCV(rfc, param_distributions=random_forest_grid, scoring='f1', n_iter=100)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)
y_pp_rf = rf_model.predict_proba(X_test)

# Report Results
print("\nRANDOM FOREST METRICS")
print("Model recall: {:.3f}".format(recall_score(y_test, y_pred_rf)))
print("Model precision: {:.3f}".format(precision_score(y_test, y_pred_rf)))
print("Model f1: {:.3f}".format(f1_score(y_test, y_pred_rf)))
print("Model accuracy: {:.3f}".format(rf_model.score(X_test, y_test)))

print ("\nCONFUSION MATRIX")
print (confusion_matrix(y_test, y_pred_rf))
print ("\nkey:")
print (" TN   FP ")
print (" FN   TP ")


print("\n")
print(datetime.datetime.now())
print("Combine The Models")

# combine the existing ones
results = pd.DataFrame(
    {'y_test': y_test,
     'rf': y_pred_rf,
     'gb': y_pred_gb,
     'ab': y_pred_ab,
     "rf_pp": y_pp_rf[:,1],
     "gb_pp": y_pp_gb[:,1],
     "ab_pp": y_pp_ab[:,1]
    })

# create new features
results['s'] = results.ab+results.gb+results.rf
results["s_pp"] = results.ab_pp+results.gb_pp+results.rf_pp
results['p'] = results['s'].apply(lambda x: 1 if x > 0 else 0)

results.to_pickle('ensemble_results.pkl')

filename = 'ensemble_results'+str(random.randint(1, 1000000))+'.pkl'
results.to_pickle(filename)
print("saved as ", filename)

print("\n")
print(datetime.datetime.now())
print("All Done!")

# # determine cutoffs
# f1 = []
# precision = []
# recall = []
# for cutoff in np.arange(0, 3.1, .01):
#     results['p_pp'] = results['s_pp'].apply(lambda x: 1 if x > cutoff else 0)
#     f1.append(f1_score(y_test, results['p_pp']))
#     precision.append(precision_score(y_test, results['p_pp']))
#     recall.append(recall_score(y_test, results['p_pp']))
