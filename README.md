# MatchingService

### 1. read in data (1_get_dataframes.ipynb)

  _INPUTS_
  - file 1: data_users.json
  - file 2: data_messages.json
  - file 3: data_cities.json

  _PY FILES_
  - file 1: my_dataframes.py
    - get_message_df()
    - get_conversation_df(message_df)
    - get_lastmessage_df()
    - get_response_df(convo_df)
    - get_user_data()
    
  - file 2: my_split.py
    - ect_find_split(df, percent)
    - ect_make_split(df, cutoff_timestamp)
  
 - file 3: my_pickle.py
    - pickle_it(df, name)
   
 _OUTPUTS_
  - file 1: data_users.pkl
  - file 2: data_convo.pkl

### 2. generate features (2_get_features.ipynb)

  _INPUTS_
   - file 1: data_users.pkl
   - file 2: data_convo.pkl 
  
 _PY FILES_
  - file 1: my_pickle.py
    - unpickle_it(name)
    - pickle_it(df, name)
