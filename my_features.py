import pandas as pd

def remove_bad_uids(df, user_df):
    '''
    removes and rows of the df that have uids not in uids
    '''
    if len(user_df.index) != len(user_df.index.unique()): return 'panic!'
    uids = set(user_df.index)
    df['flag1'] = df.first_uid.apply(lambda x: x not in uids)
    df['flag2'] = df.second_uid.apply(lambda x: x not in uids)
    df['flag'] = df.flag1 | df.flag2

    df = df.drop(df[df.flag].index)

    df = df.drop(['flag1','flag2','flag'], axis=1)
    return df


def get_rent_range(first_user, second_user):
    max1 = first_user.maxCost
    max2 = second_user.maxCost
    min1 = first_user.minCost
    min2 = second_user.minCost
    if max1*max2*min1*min2 > 0:
        upper = min(first_user.maxCost,second_user.maxCost)
        lower = max(first_user.minCost,second_user.minCost)
        if upper-lower > 0: return upper-lower
        else: return 0
    else: return 175

def test_get_rent_range():
    yes1 = '0a9yOPKFSH'
    yes2 = '013LzOrVju'
    no1 = '0UBLgJIHgz'
    no2 = '0EWNOmyQmW'
    low = '01DE0NCjwh'
    high = '02GDyQPLII'
    print(get_rent_range(user_df.loc[yes1], user_df.loc[yes2])) # should be 150
    print(get_rent_range(user_df.loc[yes1], user_df.loc[no1]))  # should be 175
    print(get_rent_range(user_df.loc[no2], user_df.loc[no1]))   # should be 175
    print(get_rent_range(user_df.loc[low], user_df.loc[high]))  # should be 0

def get_average_overlap(df,samples):
    # to find average overlap of two users' rent ranges
    # comes out to about $175
    s = 0
    for n in range(samples):
        two_users = choice(user_df[user_df.cost_range>0].index,2)
        overlap = user_df.loc[two_users,'maxCost'].min()-user_df.loc[two_users,'minCost'].max()
        if overlap < 0: overlap = 0
        s += overlap
    return s/samples


def get_inverse_distance(first_user, second_user):
    x1 = first_user.longitude
    x2 = second_user.latitude
    y1 = first_user.longitude
    y2 = second_user.latitude
    return ((x2-x1)**2+(y2-y1)**2)**(-.5)

def roommate_rules(roommates):
    if roommates > 4: return 3
    elif roommates > 1: return 2
    elif roommates > 0: return 1
    else: return 0

def get_similar_roommates(first_user, second_user):
    roommates1 = roommate_rules(first_user.numRoommates)
    roommates2 = roommate_rules(second_user.numRoommates)
    return abs(roommates1-roommates2)

def feature_time(df, user_df):
    '''
    age_dif: difference in ages between users
    rent_overlap: buy how much do their ideal rent ranges overlap
    same_gender: m/m or f/f
    same_relate:  are they both in relationships or single?
    same_clean: are the both clean/messy
    same_night: are they both early-birds or night owls?
    same_student: are they both students?
    sender_attractiveness: do people generally respond to this senders messages?
    receiver_selectivity: does this receiver generally respond to peopls messages?
    '''
    df = remove_bad_uids(df, user_df)

    ad,ro,di,rn,sg,sr,sc,sn,ss,sm,st,sa,rs = [],[],[],[],[],[],[],[],[],[],[],[],[]

    for index, row in df.iterrows():
        first = user_df.loc[str(row.first_uid)]
        second = user_df.loc[str(row.second_uid)]
        ad.append(abs(first.age - second.age))
        ro.append(get_rent_range(first, second))
        # di.append(get_inverse_distance(first, second))
        rn.append(get_similar_roommates(first, second))
        sg.append(first.gender == second.gender)
        sr.append(first.inRelationship == second.inRelationship)
        sc.append(first.isClean == second.isClean)
        sn.append(first.isNight == second.isNight)
        ss.append(first.isStudent == second.isStudent)
        sm.append(first.smokingOk == second.smokingOk)
        st.append(first.term == second.term)
        sa.append(first.attractiveness)
        rs.append(second.selectivity)

    print(len(df))
    print(len(sr))
    df['age_dif'] = ad
    df['rent_overlap'] = ro
    # df['inverse_distance'] = di
    df['roommate_num_sim'] = rn
    df['same_gender'] = sg
    df['same_relate'] = sr
    df['same_clean'] = sc
    df['same_night'] = sn
    df['same_student'] = ss
    df['same_smoking'] = sm
    df['same_term'] = st
    df['sender_attractiveness'] = sa
    df['receiver_selectivity'] = rs

    return df
