import pandas as pd

def feature_time(df, user_df):
    ad,sg,sr,sc,sn,ss,t = [],[],[],[],[],[],[]

    c=0
    for index, row in df.iterrows():
        try:
            first = user_df.loc[str(row.first_uid)]
            second = user_df.loc[str(row.second_uid)]
            ad.append(abs(first.age - second.age))
            sg.append(first.gender == second.gender)
            sr.append(first.inRelationship == second.inRelationship)
            sc.append(first.isClean == second.isClean)
            sn.append(first.isNight == second.isNight)
            ss.append(first.isStudent == second.isStudent)
            t.append(False)
        except:
            ad.append('trouble')
            sg.append('trouble')
            sr.append('trouble')
            sc.append('trouble')
            sn.append('trouble')
            ss.append('trouble')
            t.append(True)

            c += 1

    df['age_dif'] = ad
    df['same_gender'] = sg
    df['same_relate'] = sr
    df['same_clean'] = sc
    df['same_night'] = sn
    df['same_student'] = ss
    df['trouble'] = t

    df = df.drop(df[df.trouble].index)
    df = df.drop(['trouble'], axis=1)
    print("Num bad rows: {}".format(c))
    return df
