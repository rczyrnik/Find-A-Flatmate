'''
documentation!
'''

import pandas as pd
import json

def pickle_it(df,name):
    filepath = '/Users/gandalf/Documents/data/'+name+'.pkl'
    df.to_pickle(filepath)

def unpickle_it(name):
    filepath = '/Users/gandalf/Documents/data/'+name+'.pkl'
    return pd.read_pickle(filepath)

def json_it(df,name):
    filepath = '/Users/gandalf/Documents/data/'+name+'.json'
    df.to_json(filepath)

def unjson_it(name):
    filepath = '/Users/gandalf/Documents/data/'+name+'.json'
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
