import pandas as pd
import numpy as np

# import my_pickle as mp
# import my_resample as ms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from sklearn.model_selection import GridSearchCV

def oversample(X, y, tp):
    """Randomly choose positive observations from X & y, with replacement
    to achieve the target proportion of positive to negative observations.

    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - range [0, 1], target proportion of positive class observations

    Returns
    -------
    X_undersampled : ndarray - 2D
    y_undersampled : ndarray - 1D
    """
    if tp < np.mean(y):
        return X, y
    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    positive_range = np.arange(pos_count)
    positive_size = (tp * neg_count) / (1 - tp)
    positive_idxs = np.random.choice(a=positive_range,
                                     size=int(positive_size),
                                     replace=True)
    X_positive_oversampled = X_pos[positive_idxs]
    y_positive_oversampled = y_pos[positive_idxs]
    X_oversampled = np.vstack((X_positive_oversampled, X_neg))
    y_oversampled = np.concatenate((y_positive_oversampled, y_neg))

    return X_oversampled, y_oversampled

# def get_data():
#     '''
#     returns x then y
#     '''
#     return mp.unjson_it('data_X'), mp.unjson_it('data_y')['response']
def div_count_pos_neg(X, y):
    """Helper function to divide X & y into positive and negative classes
    and counts up the number in each.
    Parameters
    ----------
    X : ndarray - 2D
    y : ndarray - 1D
    Returns
    -------
    negative_count : Int
    positive_count : Int
    X_positives    : ndarray - 2D
    X_negatives    : ndarray - 2D
    y_positives    : ndarray - 1D
    y_negatives    : ndarray - 1D
    """
    negatives, positives = y == 0, y == 1
    negative_count, positive_count = np.sum(negatives), np.sum(positives)
    X_positives, y_positives = X[positives], y[positives]
    X_negatives, y_negatives = X[negatives], y[negatives]
    return negative_count, positive_count, X_positives, \
           X_negatives, y_positives, y_negatives

def prepare_data(X, y):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(), random_state=17)

    # resample
    X_train, y_train = oversample(X_train, y_train, .5)

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
    # X,y = get_data()

    # on ec2:
    X = pd.read_json('data_X.json')
    y = pd.read_json('data_y.json')['response']

    X_train, X_test, y_train, y_test = prepare_data(X, y)
    model = GradientBoostingClassifier()
    # recall, precision, accuracy = fit_data(model, X_train, X_test, y_train, y_test)
    #
    # print(recall, precision, accuracy)
    #

    GBC_grid = {
                # 'loss' : ['deviance', 'exponential'],
                'learning_rate': [.1,.4,.8],
                # 'n_estimators': [100],
                'max_depth': [3,5,10],
                # 'criterion': ['friedman_mse','mse','mae'],
                'min_samples_split': [.2,.7,2,3],
                'min_samples_leaf': [1,2,3],
                'subsample': [.2,.5,1],
                # 'max_features': ['auto','sqrt','log2',None]
                }
    #                 max_leaf_nodes
    #                 min_impurity_split
    #                 min_impurity_decrease
    #
    # param_grid = {'max_features' : [None],
    #               'n_estimators' : [10,20,30],
    #               'max_depth': [50],
    #               'min_samples_leaf': [3]
    #               }


    # GBC_grid = {'learning_rate': [.1,.4,.6,.7,.8,.9,1],
    #             'max_depth': [1, 3, 5],}
    grid_search = GridSearchCV(model, GBC_grid, n_jobs=-1).fit(X_train, y_train)

    print(grid_search.best_score_, grid_search.best_params_)
