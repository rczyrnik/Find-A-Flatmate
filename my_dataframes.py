import pandas as pd
import json
from collections import defaultdict


def get_user_data():
    filename = "/Users/gandalf/Documents/data/data_users.csv"

    user_df = pd.read_csv(filename, parse_dates=['created',
                                            'updated',
                                            'available',
                                            'birthday',
                                            'lastActive'])

    user_df = user_df.fillna({'about'    : ''})
#                     'birthday' : pd.to_datetime('1899-01-01'),
#                     'latitude' : 0,
#                     'longitude': 0})

    user_df['len_about'] = user_df.about.apply(lambda x: len(x))
    user_df['has_about'] = user_df.len_about > 0
    user_df['age'] = 2018-user_df['birthday'].apply(lambda x: x.year)

    col_to_drop = ['id']
    user_df = user_df.drop(col_to_drop, axis=1)

    user_df = user_df[user_df.onboarded==1]

    user_df = user_df[['uid', 'created', 'updated', 'about','has_about', 'len_about', 'available',
           'birthday', 'age', 'collegeId', 'emailVerified', 'foundRoommate', 'gender',
           'groupChat', 'hometownId', 'inRelationship', 'isClean', 'isNight',
           'isStudent', 'lastActive', 'latitude', 'longitude', 'maxCost',
           'minCost', 'numRoommates', 'onboarded', 'petsOk', 'pictureId',
           'roomPostId', 'roomTypeId', 'smokingOk', 'term', 'work']]


    user_df = user_df.drop_duplicates()
    user_df = user_df.set_index('uid')
    return user_df

def get_messages():
    filename = "/Users/gandalf/Documents/data/data_messages.json"
    json_data=open(filename).read()
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
    c = 0
    for conversation_id, conversation_data in data["conversations"].items():
        for message_id, message_data in conversation_data.items():
            information = defaultdict(lambda: '', message_data)
            ci.append(conversation_id)
            mi.append(message_id)
            ui.append(information['uid'])
#             if information['uid'] == conversation_id[:10]:
#                 si.append(conversation_id[10:])
#             elif information['uid'] == conversation_id[10:]:
#                 si.append(conversation_id[:10])
#             else: c += 1
            si.append(None)
            rd.append(information['read'])
            rb.append(information['readBy'])
            tl.append(len(information['text'].split()))
            ts.append(information['timestamp'])
            iu.append(information['imageURL'])
            ea.append(information['emailAttempted'])
            em.append(information['emailed'])
    print(c)
    df = pd.DataFrame([mi,ci,ui,si,rd,rb,tl,ts,iu,ea,em]).T
    df.columns=column_names

    df['date'] = df.timestamp.apply(lambda x: pd.to_datetime(x*1000000))

    df = df[['message_id', 'conversation_id', 'uid','s_uid', 'read', 'readBy', 'text_length',
           'timestamp','date', 'imageURL', 'emailAttempted', 'emailed']]


        # going to make an assumption here, and fix later if needed
#     for index, row in df.iterrows()
#         if row.first_uid == row.first_ten:
#             row.second_uid = row.last_ten
#         elif row.first_uid == row.last_ten:
#             row.second_uid = row.first_ten
#     else:
#         print("panic at the disco!")
    return df

def get_conversations(message_df):
    filename = "/Users/gandalf/Documents/data/data_messages.json"
    json_data=open(filename).read()
    data = json.loads(json_data)

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
    c = 0
    for index, row in message_df.iterrows():
        if convo_length[row.conversation_id] == 1:
            temp = row.conversation_id
            ci.append(temp)
            rs.append(False)
            fu.append(row.uid)
            fm.append(row.message_id)
#             if information['uid'] == conversation_id[:10]:
#                 su.append(conversation_id[10:])
#             elif information['uid'] == conversation_id[10:]:
#                 su.append(conversation_id[:10])
#             else: c += 1
            su.append(None)
            sm.append(None)
            ts.append(row.timestamp)

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

    df = pd.DataFrame([ci,rs,fu,su,fm,sm,ts]).T
    df.columns=column_names

    df['first_ten'] = df.conv_id.apply(lambda x: x[:10])
    df['last_ten'] = df.conv_id.apply(lambda x: x[10:])

    df=df[24:]

    # going to make an assumption here, and fix later if needed
    c = 0
    for index, row in df.iterrows():
        if row.first_ten == row.first_uid: row.second_uid = row.last_ten
        elif row.last_ten == row.first_uid: row.second_uid = row.first_ten
        else:c += 1

    df = df.drop(['first_ten', 'last_ten'], axis=1)

    print("if this number is not zero, you have a problem: {}".format(c))

    df = df.sort_values(['timestamp'])
#     df=df.reset_index()
    return df

def get_lastmessage():
    filename = "/Users/gandalf/Documents/data/data_messages.json"
    json_data=open(filename).read()
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

    return lastmessage_df

def get_responsiveness(convo_df):
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
    selectivity = 1-responses_sent/messages_received

    user_response = pd.concat([messages_sent, responses_received, attractiveness,
                               messages_received, responses_sent, selectivity], axis=1)
    user_response.columns=[['messages_sent', 'responses_received', 'attractiveness',
                               'messages_received', 'responses_sent', 'selectivity']]
    user_response = user_response.fillna(0)

    return user_response
