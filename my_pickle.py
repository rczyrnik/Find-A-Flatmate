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
