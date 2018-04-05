import pandas as pd

def roommate_rules(roommates):
    if roommates > 4: return 3
    elif roommates > 1: return 2
    elif roommates > 0: return 1
    else: return 0

def get_rent_range(row):
    max1 = row.maxCost_sender
    max2 = row.maxCost_receiver
    min1 = row.minCost_sender
    min2 = row.minCost_receiver
    if max1*max2*min1*min2 > 0:
        upper = min(max1,max2)
        lower = max(min1,min2)
        if upper-lower > 0: return upper-lower
        else: return 0
    else: return 175

def my_distance(row):
    a = row.location_receiver
    b = row.location_sender
    try:
        one = b[0]-a[0]
        two = b[1]-a[1]
        return (one**2+two**2)**(.5)
    except:
        return None

def get_features(df):
    # age difference
    df['age_dif'] = abs(df.age_sender-df.age_receiver)

    # similarities
    df['same_gender'] = df.gender_sender==df.gender_receiver
    df['same_relate'] = df.inRelationship_sender == df.inRelationship_receiver
    df['same_clean'] = df.isClean_sender == df.isClean_receiver
    df['same_night'] = df.isNight_sender == df.isNight_receiver
    df['same_student'] = df.isStudent_sender == df.isStudent_receiver
    df['same_smoking'] = df.smokingOk_sender == df.smokingOk_receiver
    df['same_type'] = df.type_sender == df.type_receiver
    df['same_term'] = df.term_sender == df.term_receiver
    df['same_work'] = df.work_sender == df.work_receiver
    df['same_city'] = df.hometownCity_sender == df.hometownCity_receiver
    df['same_state'] = df.hometownState_sender == df.hometownState_receiver
    df['same_country'] = df.hometownCountry_sender == df.hometownCountry_receiver
    df['same_college'] = df.college_sender == df.college_receiver
    df['same_metro'] = df.metro_sender == df.metro_receiver

    # overlaps
    df['overlap_roommate'] = abs(df.numRoommates_sender.apply(lambda x: roommate_rules(x))
                                  -df.numRoommates_receiver.apply(lambda x: roommate_rules(x)))
    df['hobbies_receiver'] = df.hobbies_receiver.apply(lambda x: set(x) if isinstance(x,list) else set())
    df['hobbies_sender'] = df.hobbies_sender.apply(lambda x: set(x) if isinstance(x,list) else set())
    df['overlap_hobbies'] = df.apply(lambda x: len(x['hobbies_receiver'].intersection(x['hobbies_sender'])), axis=1)

    df['amenities_receiver'] = df.amenities_receiver.apply(lambda x: set(x) if isinstance(x,list) else set())
    df['amenities_sender'] = df.amenities_sender.apply(lambda x: set(x) if isinstance(x,list) else set())
    df['overlap_amenities'] = df.apply(lambda x: len(x['amenities_receiver'].intersection(x['amenities_sender'])), axis=1)

    df['neighborhoods_receiver'] = df.neighborhoods_receiver.apply(lambda x: set(x) if isinstance(x,list) else set())
    df['neighborhoods_sender'] = df.neighborhoods_sender.apply(lambda x: set(x) if isinstance(x,list) else set())
    df['overlap_neighborhoods'] = df.apply(lambda x: len(x['neighborhoods_receiver'].intersection(x['neighborhoods_sender'])), axis=1)

    df['overlap_rent'] = df.apply(get_rent_range, axis=1)

    # urgencies
    df['urgency_receiver'] = df.available_receiver-df.timestamp
    df.urgency_receiver = df.urgency_receiver.apply(lambda x: x.days)

    df['urgency_sender'] = df.available_sender-df.timestamp
    df.urgency_sender = df.urgency_sender.apply(lambda x: x.days)

    # distance between sender and receiver
    df['distance'] = df.apply(my_distance, axis=1)


    # rename T/F as 1/0
    binary = {True: 1, False: 0}
    col_to_binary = ['response',
                 'same_work','same_city','same_state','same_country','same_metro',
                 'same_college','same_gender','same_relate','same_clean','same_night',
                 'same_student','same_smoking','same_term','same_type']
    for col in col_to_binary: df[col] = df[col].map(binary)

    print("columns with null values: {}".format(len(df.columns[df.isnull().any()])))
    return df
