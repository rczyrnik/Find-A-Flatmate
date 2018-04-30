import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def smooth(lst, times=1):
    once = [lst[0]] + [.25*lst[x-1]+.5*lst[x]+.25*lst[x+1] for x in range(0, len(lst)-1)]

    if times == 1:
        return once
    else:
        return (smooth(once, times-1))
        
def display_metrics(y_pred, y_test):

    print("\nMETRICS")
    print("Model recall: {:.3f}".format(recall_score(y_test, y_pred)))
    print("Model precision: {:.3f}".format(precision_score(y_test, y_pred)))
    print("Model f1: {:.3f}".format(f1_score(y_test, y_pred)))
    print("Model accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))

    print ("\nCONFUSION MATRIX        key")
    matrix = confusion_matrix(y_test, y_pred)
    labels = [["TP", "FP"],["FN","TP"]]

    for x in [0,1]:
        print("{}\t{}\t\t{}\t{}".format(matrix[x][0], matrix[x][1],labels[x][0], labels[x][1]))

def load_arrays(scale = False, classification = True):

    data_file_path = "/Users/gandalf/Documents/coding/Galvanize/MatchingService/data/"

    cols = np.load(data_file_path+'cols.npy')

    if classification:
        X_train = np.load(data_file_path+'X_train_class.npy')
        X_test = np.load(data_file_path+'X_test_class.npy')
        y_train = np.load(data_file_path+'y_train_class.npy')
        y_test = np.load(data_file_path+'y_test_class.npy')
    else:
        X_train = np.load(data_file_path+'X_train_reg.npy')
        X_test = np.load(data_file_path+'X_test_reg.npy')
        y_train = np.load(data_file_path+'y_train_reg.npy')
        y_test = np.load(data_file_path+'y_test_reg.npy')


    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, cols

def gridsearching(model, grid, X_train, y_train):

    search = GridSearchCV(model, param_grid=grid, scoring='f1')
    search.fit(X_train, y_train)

    df = pd.DataFrame(search.cv_results_)
    df = df.drop(['params'], axis=1)

    return df

def plot_line(df, param):
    plt.plot(df.iloc[:,4], df.mean_train_score)
    plt.plot(df.iloc[:,4], df.mean_test_score)

    plt.xlabel(param)
    plt.ylabel("f1 score")

    plt.show()

def plot_bar(df, param, w=1, rotate=False):
    plt.bar(df.iloc[:,4], df.mean_train_score, width=w, yerr=df.std_train_score)
    plt.bar(df.iloc[:,4], df.mean_test_score, width=w, yerr=df.std_train_score)

    plt.xlabel(param)
    plt.ylabel("f1 score")
    if rotate:
        plt.xticks(rotation=70)

    plt.show()

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

def display_importances_trees(model, cols):
    # show feature importances
    pd.options.display.float_format = '{:,.2f}'.format
    feature_df = pd.DataFrame([cols, model.feature_importances_]).T
    feature_df.columns = ['feature','coefficient']
    return feature_df.sort_values('coefficient', ascending=False)



def get_stats(lists):
    return [np.array(lst).mean() for lst in lists]+[np.array(lst).std() for lst in lists]

def display_values(my_dic):
    for k, v in my_dic.items():
        print("{:.1f}: {:.3f} ({:.3f}), {:.3f} ({:.3f}), {:.3f} ({:.3f})".format(k, v[0][0], v[0][1], v[1][0], v[1][1], v[2][0], v[2][1]))

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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
