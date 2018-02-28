'''
documentation!
'''

import pandas as pd
import json

start_of_path = '/Users/gandalf/Documents/data/'
start_of_path = '/Users/gandalf/Documents/coding/data_do_not_commit/'

def pickle_it(df,name):
    filepath = start_of_path+name+'.pkl'
    df.to_pickle(filepath)

def unpickle_it(name):
    filepath = start_of_path+name+'.pkl'
    return pd.read_pickle(filepath)

def json_it(df,name):
    filepath = start_of_path+name+'.json'
    df.to_json(filepath)

def unjson_it(name):
    filepath = start_of_path+name+'.json'
    return pd.read_json(filepath)

def my_other_to_datetime(x):
    # if isinstance(x, numpy.int64):
    try:
        return pd.to_datetime(x*1000000)
    except:
        return None
    # else:
    #     return None

def reinstate_date(df, col_list):
    for col in col_list:
        df[col] = df[col].apply(my_other_to_datetime)
    return df
