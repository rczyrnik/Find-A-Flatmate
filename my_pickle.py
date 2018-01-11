'''
documentation!
'''
import pandas as pd

def pickle_it(df,name):
    filepath = '/Users/gandalf/Documents/data/'+name+'.pkl'
    df.to_pickle(filepath)

def unpickle_it(name):
    filepath = '/Users/gandalf/Documents/data/'+name+'.pkl'
    return pd.read_pickle(filepath)
