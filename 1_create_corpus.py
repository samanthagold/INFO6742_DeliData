# GOAL: transform corpus into ConvoKit format 

# Packages 
from convokit import Corpus, download
from convokit import Transformer
import nltk
import os
import re
import pandas as pd
from tqdm import tqdm
import ast
from datetime import datetime


## Setup 
dir = "/Users/sammygold/Downloads/delidata/" #Please change it to your directory
outdir = "/Users/sammygold/Desktop/INFO6742/INFO6742_DeliData/"

##  Helper Functions Used Throughout 
# Define function that lags a column 
def lag(df, column):
  df[f'lagged_{column}'] = df[column].shift(1)
  return(df)

# Define function that extracts final card selections
def parse_meta(val):
  val = eval(val)
  return ''.join([x['value'] for x in val if x['checked']])

# Defining function that checks whether or not a final answer is correct 
def correct_flag(val):
  length = len(val)
  numbers = [int(s) for s in re.sub("\\D", "", val) if s.isdigit()]
  num_numbers = len(numbers)
  letters = [s for s in re.sub("[^a-zA-Z]+", "", val) if s.isalpha()]
  num_letters = len(letters)
  if num_numbers == 1 and num_letters == 1 and length == 2:
    odd = [x % 2 != 0 for x in numbers]
    vowel = [x.upper() in ['A', 'E', 'I', 'O', 'U'] for x in letters]
    if odd[0] and vowel[0]:
      return True
    else:
      return False
  else:
    return False
#%%
#########################################
############### DATA PREP ###############
#########################################

# Open all raw DeliData files
#filenames = [file.removesuffix('.tsv') for file in os.listdir(f'{dir}all/')]
filenames = os.listdir(f'{dir}all/') #2022-Sep-5 Jinsook: my python version is 3.8 which doesn't have removesuffix so I changed little
delidata_all= []

for filename in tqdm(filenames):
  df = pd.read_table(f'{dir}all/{filename}')
  df['convo_id'] = filename
  delidata_all.append(df)

delidata_all = pd.concat(delidata_all)
delidata_all.columns = delidata_all.columns.str.lower()

#%%
#########################################
######### UTTERANCE GENERATION ##########
#########################################

# Extract utterances
deli_utt = delidata_all[delidata_all['message_type']=='CHAT_MESSAGE']
deli_utt = deli_utt[deli_utt['user_name']!='Moderating Owl']
print(len(deli_utt)) #14,035

# Remove unnecassary signs and extract strings
deli_content_processed = []
for i in range(len(deli_utt)):
  if "{'message'" in deli_utt['content'].iloc[i]:
    a = ast.literal_eval(deli_utt['content'].iloc[i])['message']
  elif isinstance(deli_utt['content'].iloc[i], dict):
    a = deli_utt['content'].iloc[i]['message']
  else:
    a = deli_utt['content'].iloc[i]
  deli_content_processed.append(a)

# Append to deli_utt
deli_utt['text'] = deli_content_processed
deli_utt = deli_utt.drop(['content'],axis=1)

#Select required columns
deli_utt = deli_utt[['message_id', 'user_name', 'user_id','timestamp','convo_id','text']]\
  .rename(columns={'user_id':'speaker','convo_id':'conversation_id'})\
  .reset_index(drop=True)


#########################################
######## POPULATE REPLY_TO FIELD ########
#########################################

# Convert timestamp to datetime
deli_utt['timestamp2'] = pd.to_datetime(deli_utt['timestamp'])

# Sort timestamp, by convo to get correct conversation ordering
deli_utt = (
    deli_utt
    .sort_values(['timestamp2'], ascending = True)
    .groupby('conversation_id')
    .apply(lag, column = 'message_id')
    .sort_values(['conversation_id', 'timestamp2'], ascending = True)
    .rename(columns={'lagged_message_id':'reply_to', 'message_id': 'id'}, inplace = False)

)

#Subsetting and sorting relevant columns 
deli_utt = deli_utt[['id', 'timestamp', 'text', 'speaker', 'reply_to', 'conversation_id']]

#########################################
######### CREATING FINAL CORPUS #########
#########################################

# PART 1: User Level# 
# Filtering to all wason submit actions to extract final submission and 
# generating user-level data frame 
delidata_user = delidata_all[delidata_all['message_type'] == "WASON_SUBMIT"]
# Extracting final answer 
delidata_user['meta.finalanswer']=delidata_user['content'].apply(parse_meta)
# Grabbing latest submission 
delidata_user = (
    delidata_user
    .sort_values(['timestamp'], ascending = True)
    .groupby(['convo_id', 'user_id'])
    .tail(1)
)
user_df = delidata_user[['user_id', 'meta.finalanswer']].rename(columns={'user_id':'id'})
# Checking correct-ness of final answer column 
user_df['meta.correct_flag'] = user_df['meta.finalanswer'].apply(correct_flag)

# PART 2: Conversation Level#  
convo_prep = (
    deli_utt
    .groupby('conversation_id')
    .agg({'id': 'size', 'speaker': 'nunique'})
    .reset_index()
    .rename(columns={'index': 'conversation_id','id':'meta.num_chats', 'speaker':'meta.num_participants'})
)

convo_prep2 = pd.merge(deli_utt[['speaker', 'conversation_id']].drop_duplicates(), user_df, how = 'inner', left_on = 'speaker', right_on = 'id')
convo_df = (
    convo_prep2
    .groupby('conversation_id')
    .agg({'meta.correct_flag': 'sum', 'speaker':'nunique'})
    .reset_index()
    .rename(columns={'index': 'conversation_id'})
    .merge(convo_prep, how = 'inner', on = 'conversation_id')
    .rename(columns={'conversation_id':'id'})
    
)
# Generating ratio 
convo_df['meta.correct_ratio']=convo_df['meta.correct_flag']/convo_df['meta.num_participants']
# Column clean up 
convo_df = convo_df[['id', 'meta.num_chats', 'meta.num_participants', 'meta.correct_ratio']]

# Create final corpus
delidata_corpus = Corpus.from_pandas(utterances_df = deli_utt, speakers_df = user_df, conversations_df = convo_df)

# Save Corpus
delidata_corpus.dump(name="delidata_corpus", base_path=outdir)
