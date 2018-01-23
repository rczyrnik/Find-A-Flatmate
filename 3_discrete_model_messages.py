import pandas as pd
import numpy as np

import my_pickle as mp
import my_resample as ms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from sklearn.model_selection import GridSearchCV


def get_data():
    '''
    returns x then y
    '''
    return mp.unjson_it('data_X'), mp.unjson_it('data_y')['response']

def prepare_data(X, y):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(), random_state=17)

    # resample
    X_train, y_train = ms.oversample(X_train, y_train, .5)

    # scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def fit_data(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return recall_score(y_test, y_pred), precision_score(y_test, y_pred), model.score(X_test, y_test)

if __name__ == "__main__":
    # on mac:
    X,y = get_data()

    # on ec2:
    # X = pd.read_json('../data_X.json')
    # y = pd.read_json('../data_y.json')

    X_train, X_test, y_train, y_test = prepare_data(X, y)
    model = GradientBoostingClassifier()
    # recall, precision, accuracy = fit_data(model, X_train, X_test, y_train, y_test)
    #
    # print(recall, precision, accuracy)
    #

    # GBC_grid = {'loss' : [‘deviance’, ‘exponential’],
    #                 'learning_rate': [.1,.4,.8],
    #                 'n_estimators': [100],
    #                 'max_depth': [1, 3, 5],
    #                 'criterion': ['friedman_mse','mse','mae'],
    #                 'min_samples_split': [1,2,3],
    #                 'min_samples_leaf': [1, 2, 3],
    #                 'subsample': [.2,.5.1],
    #                 'max_features': ['auto','sqrt','log2',None]}
    #                 max_leaf_nodes
    #                 min_impurity_split
    #                 min_impurity_decrease
    #
    # param_grid = {'max_features' : [None],
    #               'n_estimators' : [10,20,30],
    #               'max_depth': [50],
    #               'min_samples_leaf': [3]
    #               }


    GBC_grid = {'learning_rate': [.1,.4,.8]}
    grid_search = GridSearchCV(model, GBC_grid, n_jobs=-1).fit(X_train, y_train)

    print(grid_search.best_score_, grid_search.best_params_)
