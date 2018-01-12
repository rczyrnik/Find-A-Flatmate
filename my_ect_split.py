import pandas as pd
# instructions for split
def ect_find_split(df, percent):
    cutoff_index = int(len(df)*percent)-1
    return df.loc[cutoff_index,'timestamp']

def ect_make_split(df, cutoff_timestamp):
    new_df = df[df.timestamp <= cutoff_timestamp]
    return new_df
