import pandas as pd
import numpy as np
import json
from collections import defaultdict
import my_pickle as mp
import my_split as ms

def get_message_df():

    # read from json
    filename = "/Users/gandalf/Documents/data/raw_data_messages.json"
    with open(filename) as f:
        json_data = f.read()
        data = json.loads(json_data)


    column_names = ['message_id',       # Added, unique for each row
                   'conversation_id',   # Added, identifies the conversation, not unique
                   'uid',               # message sender
                    's_uid',             # message receiver
                   'read',              # true/false value
                   'readBy',            # message recipient(s)
                   'text_length',       # Added, length of text
                   'timestamp',         # time
                   'imageURL',          # image url
                   'emailAttempted',    # ?
                   'emailed']           # ?

    mi,ci,ui,si,rd,rb,tl,ts,iu,ea,em = [],[],[],[],[],[],[],[],[],[],[]

    for conversation_id, conversation_data in data["conversations"].items():
        for message_id, message_data in conversation_data.items():
            information = defaultdict(lambda: '', message_data)
            ci.append(conversation_id)
            mi.append(message_id)
            ui.append(information['uid'])
            si.append(None)
            rd.append(information['read'])
            rb.append(information['readBy'])
            tl.append(len(information['text'].split()))
            ts.append(information['timestamp'])
            iu.append(information['imageURL'])
            ea.append(information['emailAttempted'])
            em.append(information['emailed'])

    # turn lists into a dataframe
    df = pd.DataFrame([mi,ci,ui,si,rd,rb,tl,ts,iu,ea,em]).T
    df.columns=column_names

    # convert to datetime
    df['date'] = df.timestamp.apply(lambda x: pd.to_datetime(x*1000000))

    # rearrange columns
    df = df[['message_id', 'conversation_id', 'uid','s_uid', 'read', 'readBy', 'text_length',
           'timestamp','date', 'imageURL', 'emailAttempted', 'emailed']]

    df['flag'] = df.timestamp.apply(lambda x: not isinstance(x, int))

    df = df.drop(df[df.flag].index)

    print("created message dataframe")
    return df

def get_conversation_df(message_df):
    # filename = "/Users/gandalf/Documents/data/data_messages.json"
    # json_data=open(filename).read()
    # data = json.loads(json_data)

    message_df['const'] = 1
    convo_length = message_df.groupby('conversation_id').const.sum().T.to_dict()

    column_names = ['conv_id',       # user id
                    'response',      # did anyone respond?
                    'uid_sender',     # user who sent the first message
                    'uid_receiver',    # user who responded
                    'len_sender',     # message id of the first message
                    'len_receiver',    # message id of the second message
                    'mid_sender',     # message id of the first message
                    'mid_receiver',    # message id of the second message
                    'timestamp'
                   ]

    ci,rs,fu,fl,fm,su,sl,sm,ts = [],[],[],[],[],[],[],[],[]
    already_added = set()

    first_message = True
    second_message = False

    for index, row in message_df.iterrows():
        # if there was no response
        if convo_length[row.conversation_id] == 1:
            temp = row.conversation_id
            ci.append(temp)
            rs.append(False)
            fu.append(row.uid)
            fl.append(row.text_length)
            fm.append(row.message_id)
            su.append(None)
            sl.append(None)
            sm.append(None)
            ts.append(row.timestamp)

        # if there was a response
        elif row.conversation_id not in already_added:
            if first_message:
                ci.append(row.conversation_id)
                rs.append(True)
                fu.append(row.uid)
                fm.append(row.message_id)
                first_message = False
                second_message = True
                ts.append(row.timestamp)
            elif second_message:
                su.append(row.uid)
                sl.append(row.text_length)
                sm.append(row.message_id)
                already_added.add(row.conversation_id)
                first_message = True
                second_message = False

    # create dataframe from lists
    df = pd.DataFrame([ci,rs,fu,su,fl,sl,fm,sm,ts]).T
    df.columns=column_names

    # get two userids
    df['first_ten'] = df.conv_id.apply(lambda x: x[:10])
    df['last_ten'] = df.conv_id.apply(lambda x: x[10:])

    # get rid of the first 24 rows because they are trouble makers
    df=df[24:]

    # going to make an assumption here, and fix later if needed
    c = 0
    for index, row in df.iterrows():
        if row.first_ten == row.uid_sender: row.uid_receiver = row.last_ten
        elif row.last_ten == row.uid_sender: row.uid_receiver = row.first_ten
        else:c += 1

    df = df.drop(['first_ten', 'last_ten'], axis=1)

    df = df.sort_values(['timestamp'])

    print("created conversation dataframe with {} known errors".format(c))

    return df

def get_lastmessage_df():

    # read from json
    filename = "/Users/gandalf/Documents/data/raw_data_messages.json"
    with open(filename) as f:
        json_data = f.read()
        data = json.loads(json_data)

    column_names = ['user_id',       # user id
                   'first_ten',      # first ten of the 20 digit key
                   'last_ten',       # last ten of the 20 digit key (prob userid)
                   'lastMessageId']  # message id

    ui,ft,lt,lm = [],[],[],[]

    for user_id, user_data in data['users'].items():
        for key, value in user_data['conversations'].items():
            ui.append(user_id)
            ft.append(key[:10])
            lt.append(key[10:])
            lm.append(value['lastMessageId'])

    lastmessage_df = pd.DataFrame([ui,ft,lt,lm]).T
    lastmessage_df.columns=column_names

    print("created last message dataframe")

    return lastmessage_df

def get_response_df(convo_df):
    # Get responsiveness table
    convo_df['const'] = 1

    # how many did he send?
    messages_sent = convo_df.groupby('first_uid').const.sum()
    # how many resposnes did he get?
    responses_received = convo_df[convo_df.response == True].groupby('first_uid').const.sum()

    # ratio (higher = more attractive roommmate)
    attractiveness = responses_received/messages_sent

    # how many did he receive?
    messages_received = convo_df.groupby('second_uid').const.sum()
    # how many did he respond to?
    responses_sent = convo_df[convo_df.response == True].groupby('first_uid').const.sum()
    # ratio (higher = responds less)
    responsiveness = responses_sent/messages_received

    # combine into a dataframe and name columns
    user_response = pd.concat([messages_sent, responses_received, attractiveness,
                               messages_received, responses_sent, responsiveness], axis=1)
    user_response.columns=[['messages_sent', 'responses_received', 'attractiveness',
                               'messages_received', 'responses_sent', 'responsiveness']]

    # get averages
    average_attractiveness = user_response[user_response.attractiveness >0].mean()['attractiveness']
    average_responsiveness = user_response[user_response.responsiveness >0].mean()['responsiveness']

    # deal with NaN on attractiveness side
    user_response = user_response.fillna({'messages_sent':0, 'responses_received':0})
    for index, row in user_response.iterrows():

        if row.messages_sent == 0: row.attractiveness = average_attractiveness
        elif row.responses_received == 0: row.attractiveness = 0
    # user_response = user_response.fillna({'attractiveness':0})

    # deal with NaN on responsiveness side
    user_response = user_response.fillna({'messages_received':0, 'responses_sent':0})
    for index, row in user_response.iterrows():
        if row.messages_received == 0: row.responsiveness = average_responsiveness
        if row.responses_sent == 0: row.responsiveness = 0
    # user_response = user_response.fillna({'attractiveness':0})

    return user_response

def my_to_datetime(x):
    if isinstance(x, dict):
        try:
            return pd.to_datetime(x['$date'])
        except:
            return None
    else:
        return None

def get_hoods(lst):
    if isinstance(lst, list):
        temp_set = set()
        for thing in lst:
            temp_set.add(thing['objectId'])
        return temp_set
    else:
        return set()

def get_user_data():

    # read in json as dataframe
    filename = "/Users/gandalf/Documents/data/raw_data_users.json"
    df = pd.read_json(filename)


    df = df.rename(index=str, columns={"_created_at": "created",
                                       "_updated_at": "updated",
                                       "_p_room"    : "has_room",
                                       "_id"        : "uid"})
    df = df.set_index('uid')

    # change dates from strings to date times
    df.created = df.created.apply(lambda x: my_to_datetime(x))
    df.updated = df.updated.apply(lambda x: my_to_datetime(x))
    df.activeAt = df.activeAt.apply(lambda x: my_to_datetime(x))
    df.available = df.available.apply(lambda x: my_to_datetime(x))
    df.birthday = df.birthday.apply(lambda x: my_to_datetime(x))
    df['age'] = 2018-df['birthday'].apply(lambda x: x.year)


    df.neighborhoods = df.neighborhoods.apply(get_hoods)

    # drop unused columns
    col_to_drop = ['_acl','_auth_data_facebook','_hashed_password','_rperm','_wperm','blocked','email','emailVerified','firstName',
                      'foundRoommate','groupChat','hometown','hometownCounty','likes','lastName','positions','recommended','username']
    df = df.drop(col_to_drop, axis = 1)

    # fill in na values
    df = df.fillna({'about':''})


    # create new features
    df['len_about'] = df.about.apply(lambda x: len(x))
    df['has_about'] = df.len_about > 0
    df['I_count'] = df.about.apply(lambda x: x.count('I'))
    df['I_ratio'] = df.about.apply(lambda x: x.count('I')/len(x) if len(x) > 0 else np.nan)
    df['period_count'] = df.about.apply(lambda x: x.count('.'))
    df['period_ratio'] = df.about.apply(lambda x: x.count('.')/len(x) if len(x) > 0 else np.nan)
    df['question_count'] = df.about.apply(lambda x: x.count('?'))
    df['question_ratio'] = df.about.apply(lambda x: x.count('?')/len(x) if len(x) > 0 else np.nan)
    df['exclaim_count'] = df.about.apply(lambda x: x.count('!'))
    df['exclaim_ratio'] = df.about.apply(lambda x: x.count('!')/len(x) if len(x) > 0 else np.nan)
    df['sentence_count'] = df.period_count+df.question_count+df.exclaim_count
    df['sentence_ratio'] = df.period_ratio+df.question_ratio+df.exclaim_ratio
    df.has_room = df.has_room.apply(lambda x: isinstance(x,str))
    df.facebookId = df.facebookId.apply(lambda x: isinstance(x,str))
    df.linkedinId = df.linkedinId.apply(lambda x: isinstance(x,str))
    df.picture = df.picture.apply(lambda x: isinstance(x,str))


    # reformat df by renaming and moving around columns


    # df = df[['created', 'updated', 'activeAt', 'available', # dates
    #          'about', 'has_about', 'len_about',            # about
    #          'birthday', 'age', 'gender', 'location', 'work',      # demographic
    #          'hometownCity', 'hometownCountry', 'hometownState',     # more demographic
    #          'college', 'facebookId','linkedinId', 'picture',        # engagement
    #          'maxCost', 'minCost', 'neighborhoods', 'numRoommates', 'term', 'type', 'has_room',  # room basics
    #          'amenities', 'hobbies',                                              # room not boolean
    #          'inRelationship', 'isClean', 'isNight', 'isStudent', 'petsOk', 'smokingOk',   # room boolean
    #          'onboarded']]

    df = df.drop(df[df.onboarded != 1].index)
    print("created user dataframe")

    return df

def remove_bad_uids(df, user_df):
    '''
    removes rows of the df that have uids not in uids
    '''
    # uids better be unique
    if len(user_df.index) != len(user_df.index.unique()): return 'panic!'

    # get set of all uids
    uids = set(user_df.index)

    # set flag if first or second uid not in set of uids
    df['flag1'] = df.uid_sender.apply(lambda x: x not in uids)
    df['flag2'] = df.uid_receiver.apply(lambda x: x not in uids)
    df['flag'] = df.flag1 | df.flag2

    print("{} rows dropped".format(df.flag.sum()))
    # drop rows that were flaged
    df = df.drop(df[df.flag].index)

    # drop flag columns
    df = df.drop(['flag1','flag2','flag'], axis=1)

    return df
