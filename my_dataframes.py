import pandas as pd
import json
from collections import defaultdict

def get_message_df():

    # read from json
    filename = "/Users/gandalf/Documents/data/data_messages.json"
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
                    'first_uid',     # user who sent the first message
                    'second_uid',    # user who responded
                    'first_mid',     # message id of the first message
                    'second_mid',    # message id of the second message
                    'timestamp'
                   ]

    ci,rs,fu,fm,su,sm,ts = [],[],[],[],[],[],[]
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
            fm.append(row.message_id)
            su.append(None)
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
                sm.append(row.message_id)
                already_added.add(row.conversation_id)
                first_message = True
                second_message = False

    # create dataframe from lists
    df = pd.DataFrame([ci,rs,fu,su,fm,sm,ts]).T
    df.columns=column_names

    # get two userids
    df['first_ten'] = df.conv_id.apply(lambda x: x[:10])
    df['last_ten'] = df.conv_id.apply(lambda x: x[10:])

    # get rid of the first 24 rows because they are trouble makers
    df=df[24:]

    # going to make an assumption here, and fix later if needed
    c = 0
    for index, row in df.iterrows():
        if row.first_ten == row.first_uid: row.second_uid = row.last_ten
        elif row.last_ten == row.first_uid: row.second_uid = row.first_ten
        else:c += 1

    df = df.drop(['first_ten', 'last_ten'], axis=1)

    df = df.sort_values(['timestamp'])

    print("created conversation dataframe with {} known errors".format(c))

    return df

def get_lastmessage_df():

    # read from json
    filename = "/Users/gandalf/Documents/data/data_messages.json"
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

def get_user_data(response_df):
    # filename = "/Users/gandalf/Documents/data/data_users.csv"

    # user_df = pd.read_csv(filename, parse_dates=['created',
    #                                         'updated',
    #                                         'available',
    #                                         'birthday',
    #                                         'lastActive'], na_values='nan')
    #



    # user_df = user_df.fillna({'about'    : ''})
    #
    # user_df['len_about'] = user_df.about.apply(lambda x: len(x))
    # user_df['has_about'] = user_df.len_about > 0
    # user_df['age'] = 2018-user_df['birthday'].apply(lambda x: x.year)
    #
    # col_to_drop = ['id']
    # user_df = user_df.drop(col_to_drop, axis=1)
    #
    # user_df = user_df[user_df.onboarded==1]
    #
    # user_df = user_df[['uid', 'created', 'updated', 'about','has_about', 'len_about', 'available',
    #        'birthday', 'age', 'collegeId', 'emailVerified', 'foundRoommate', 'gender',
    #        'groupChat', 'hometownId', 'inRelationship', 'isClean', 'isNight',
    #        'isStudent', 'lastActive', 'latitude', 'longitude', 'maxCost',
    #        'minCost', 'numRoommates', 'onboarded', 'petsOk', 'pictureId',
    #        'roomPostId', 'roomTypeId', 'smokingOk', 'term', 'work']]
    #
    #
    # user_df = user_df.drop_duplicates()
    # user_df = user_df.set_index('uid')
    #
    # user_df = user_df.join(response_df)
    #
    # average_attractiveness = response_df[response_df.attractiveness > 0].mean()['attractiveness']
    # average_responsiveness= response_df[response_df.responsiveness < 1].mean()['responsiveness']
    #
    # user_df = user_df.fillna({'messages_sent' : 0,'responses_received': 0,'attractiveness':average_attractiveness,
    #                           'messages_received': 0,'responses_sent': 0, 'responsiveness':average_responsiveness})
    #

    filename = "/Users/gandalf/Documents/data/data_users.json"

    df = pd.read_json(filename)

    col_to_drop = ['_acl','_auth_data_facebook','_hashed_password','_rperm','_wperm','blocked','email','emailVerified','firstName',
                      'foundRoommate','groupChat','hometown','hometownCounty','likes','lastName','positions','recommended','username']
    df = df.drop(col_to_drop, axis = 1)

    df = df.fillna({'about':''})

    df = df.rename(index=str, columns={"_created_at": "created",
                                       "_updated_at": "updated",
                                       "_p_room"    : "has_room",
                                       "_id"        : "uid"})

    df = df.set_index('uid')
    df['len_about'] = df.about.apply(lambda x: len(x))
    df['has_about'] = df.len_about > 0
    df.has_room = df.has_room.apply(lambda x: isinstance(x,str))
    df.facebookId = df.facebookId.apply(lambda x: isinstance(x,str))
    df.linkedinId = df.linkedinId.apply(lambda x: isinstance(x,str))
    df.picture = df.picture.apply(lambda x: isinstance(x,str))



    df.created = df.created.apply(lambda x: my_to_datetime(x))
    df.updated = df.updated.apply(lambda x: my_to_datetime(x))
    df.activeAt = df.activeAt.apply(lambda x: my_to_datetime(x))
    df.available = df.available.apply(lambda x: my_to_datetime(x))
    df.birthday = df.birthday.apply(lambda x: my_to_datetime(x))
    df['age'] = 2018-df['birthday'].apply(lambda x: x.year)

    df = df[['created', 'updated', 'activeAt', 'available', # dates
             'about', 'has_about', 'len_about',            # about
             'birthday', 'age', 'gender', 'location', 'work',      # demographic
             'hometownCity', 'hometownCountry', 'hometownState',     # more demographic
             'college', 'facebookId','linkedinId', 'picture',        # engagement
             'maxCost', 'minCost', 'neighborhoods', 'numRoommates', 'term', 'type', 'has_room',  # room basics
             'amenities', 'hobbies',                                              # room not boolean
             'inRelationship', 'isClean', 'isNight', 'isStudent', 'petsOk', 'smokingOk',   # room boolean
             'onboarded',]]

    df = df.join(response_df)

    average_attractiveness = response_df[response_df.attractiveness > 0].mean()['attractiveness']
    average_responsiveness = response_df[response_df.responsiveness > 0].mean()['responsiveness']

    df = df.fillna({'messages_sent' : 0,'responses_received': 0,'attractiveness':average_attractiveness,
                              'messages_received': 0,'responses_sent': 0, 'responsiveness':average_responsiveness})


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
    df['flag1'] = df.first_uid.apply(lambda x: x not in uids)
    df['flag2'] = df.second_uid.apply(lambda x: x not in uids)
    df['flag'] = df.flag1 | df.flag2

    # drop rows that were flaged
    df = df.drop(df[df.flag].index)

    # drop flag columns
    df = df.drop(['flag1','flag2','flag'], axis=1)

    return df
