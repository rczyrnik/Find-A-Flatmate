import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def convert_to_binary(lst, cutoff=1):
    return [1 if x > cutoff else 0 for x in lst]

def display_importances_linear(model, X):
    # show feature importances
    pd.options.display.float_format = '{:,.2f}'.format
    feature_df = pd.DataFrame([X.columns, model.coef_]).T
    feature_df.columns = ['feature','coefficient']
    feature_df['abs_value'] = feature_df.coefficient.apply(abs)
    feature_df['sign'] = feature_df.coefficient/feature_df.abs_value
    return feature_df.sort_values('abs_value', ascending=False)

def display_importances_trees(model, X):
    # show feature importances
    pd.options.display.float_format = '{:,.2f}'.format
    feature_df = pd.DataFrame([X.columns, model.feature_importances_]).T
    feature_df.columns = ['feature','coefficient']
    return feature_df.sort_values('coefficient', ascending=False)

def display_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_binary = convert_to_binary(y_pred,3)
    y_test_binary = convert_to_binary(y_test,3)

    print("\nMETRICS")
    print("Model recall: {}".format(recall_score(y_test_binary, y_pred_binary)))
    print("Model precision: {}".format(precision_score(y_test_binary, y_pred_binary)))
    print("Model accuracy: {}".format(model.score(X_test, y_test)))

    print ("\nCONFUSION MATRIX")
    print (confusion_matrix(y_test_binary, y_pred_binary))
    print ("\nkey:")
    print (" TN   FP ")
    print (" FN   TP ")

    # make fake data
    # pred_all_0 = [0]*len(y_test)
    # pred_all_1 = [1]*len(y_test)
    # pred_50_50 = np.random.choice([0,1], size=len(y_test))
    # pred_90_10 = np.random.choice([0,1], size=len(y_test), p=[.9,.1])
    # print("\nRECALL AND ACCURACY FOR DIFFERNET MODELS")
    # print("recall     \t precision   \tmodel")
    # print(recall_score(y_test_binary, y_pred_binary), '\t',precision_score(y_test_binary, y_pred_binary), "my model")
    # print(recall_score(y_test_binary, pred_all_0),'\t','\t', precision_score(y_test_binary, pred_all_0), "\t\tpredict all zero")
    # print(recall_score(y_test_binary, pred_all_1),'\t','\t', precision_score(y_test_binary, pred_all_1), "predict all one")
    # print(recall_score(y_test_binary, pred_50_50),'\t', precision_score(y_test_binary, pred_50_50), "predict 50-50")
    # print(recall_score(y_test_binary, pred_90_10), precision_score(y_test_binary, pred_90_10), "predict 90-10")
