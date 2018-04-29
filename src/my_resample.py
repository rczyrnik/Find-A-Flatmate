import numpy as np

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

def undersample(X, y, tp):
   """Randomly discards negative observations from X & y to achieve the
   target proportion of positive to negative observations.

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
   negative_sample_rate = (pos_count * (1 - tp)) / (neg_count * tp)
   negative_keepers = np.random.choice(a=[False, True], size=neg_count,
                                       p=[1 - negative_sample_rate,
                                          negative_sample_rate])
   X_negative_undersampled = X_neg[negative_keepers]
   y_negative_undersampled = y_neg[negative_keepers]
   X_undersampled = np.vstack((X_negative_undersampled, X_pos))
   y_undersampled = np.concatenate((y_negative_undersampled, y_pos))

   return X_undersampled, y_undersampled


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
