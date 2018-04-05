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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

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

def gridsearch_gbc(X, y, resample_wt=.5,
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
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(), random_state=17)

    # resample
    X_train, y_train = oversample(X_train, y_train, resample_wt)

    # scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

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

    f1 = (recall*precision)/(recall + precision)
    print("resample_wt: {}, max_depth: {}, f1: {:.3},   rcll: {:.3},   prcsn: {:.3}".format(resample_wt,max_depth,f1,recall,precision))

    return recall, precision

def gridsearch_abc(X, y, resample_wt=.5, n_estimators=50, learning_rate=1):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(), random_state=17)

    # resample
    X_train, y_train = oversample(X_train, y_train, resample_wt)

    # scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = AdaBoostClassifier(n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    f1 = (recall*precision)/(recall + precision)
    print("rsmp_wt: {}, n_est: {}, lrng_rate: {}, f1: {:.3},   rcll: {:.3},   prcsn: {:.3}".format(resample_wt,n_estimators,learning_rate,f1,recall,precision))
    #
    # print("rsmp_wt: {:.2},   lrng_rt: {:.2},  mx_dpth: {}, mn_smp_sp: {}, f1: {:.3},   rcll: {:.3},   prcsn: {:.3}"
    # .format(resample_wt,learning_rate,max_depth,min_samples_split,f1,recall,precision))

    return recall, precision

if __name__ == "__main__":
    # on mac:
    # import my_pickle as mp
    # X = mp.unjson_it('data_X')
    # y = mp.unjson_it('data_y')['response']


    # on ec2:
    X = pd.read_json('data_X.json')
    y = pd.read_json('data_y.json')['response']




    # AdaBoost
    '''
    for resample_wt in np.arange(.1,.9,.05):
    for max_depth in range(1, 10):
    for min_samples_split in [.1,.3,.5,.7,.9,2,3,4]:
    for min_samples_leaf in range(1, 10):
    for subsample in np.arange(.1,1,.1):
    for max_features in range(1, len(X-1)):
    for max_leaf_nodes in [10000,100000,10000000]:
    for min_impurity_decrease in np.arange(.1,1.9, .1):
    for loss in ['deviance', 'exponential']:
    for criterion in ['friedman_mse','mse','mae']:
    '''
    r = []
    p = []

    for resample_wt in np.arange(.1,.9,.1):
        for n_estimators in [5, 10, 50, 100, 500, 1000]:
            for learning_rate in [.01, .05, .1, .5,1]:
                recall, precision  = gridsearch_abc(X, y, resample_wt=resample_wt,
                                                n_estimators=n_estimators,
                                                learning_rate=learning_rate,
                                                )
                r.append(recall)
                p.append(precision)
    print("\nrecall = {}\nprecision = {}".format(r,p))





    # GradientBoostingClassifier
    # '''
    # for resample_wt in np.arange(.1,.9,.05):
    # for max_depth in range(1, 10):
    # for min_samples_split in [.1,.3,.5,.7,.9,2,3,4]:
    # for min_samples_leaf in range(1, 10):
    # for subsample in np.arange(.1,1,.1):
    # for max_features in range(1, len(X-1)):
    # for max_leaf_nodes in [10000,100000,10000000]:
    # for min_impurity_decrease in np.arange(.1,1.9, .1):
    # for loss in ['deviance', 'exponential']:
    # for criterion in ['friedman_mse','mse','mae']:
    # '''
    #
    # r = []
    # p = []
    # for max_depth in range(1, 10):
    #     for resample_wt in np.arange(.1,.9,.05):
    #         recall, precision = gridsearch_gbc(X, y, resample_wt=resample_wt,
    #                                                 loss='exponential',
    #                                                 learning_rate=.025,
    #                                                 max_depth=max_depth,
    #                                                 min_samples_split=3,
    #                                                 min_samples_leaf=3,
    #                                                 max_features=35,
    #                                                 max_leaf_nodes=10000,
    #                                                 min_impurity_decrease=0,
    #                                                 criterion='friedman_mse',
    #                                                 )
    #         r.append(recall)
    #         p.append(precision)
    # print("\nrecall = {}\nprecision = {}".format(r,p))
