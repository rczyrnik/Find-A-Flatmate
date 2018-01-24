import pandas as pd
# instructions for split
def ect_find_split(df, percent):
    cutoff_index = int(len(df)*percent)-1
    return df.iloc[cutoff_index].loc['timestamp']

def ect_make_split(df, cutoff):
    new_df = df[df.timestamp <= cutoff]
    return new_df
