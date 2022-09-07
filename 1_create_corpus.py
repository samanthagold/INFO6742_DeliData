# GOAL: transform corpus into ConvoKit format 
## Made a few assumptions: 
  # - we want to use all raw files from delidata (not just the Mturk files
  #   or just the pilot files)

# Packages 
import convokit 
from convokit import Corpus, download 
import nltk
import os
import pandas as pd
from datetime import datetime
import ast 

# Setup 
dir = "Downloads/delidata/"

# Helpers
def lag(df, column): 
  df[f'lagged_{column}'] = df[column].shift(1)
  return(df)

############### Data Prep 
# Open all raw DeliData files
filenames = [file.removesuffix('.tsv') for file in os.listdir(f'{dir}all/')]
delidata_all= []
for filename in filenames: 
  df = pd.read_table(f'{dir}all/{filename}.tsv') 
  df['convo_id'] = filename
  delidata_all.append(df)

delidata_all = pd.concat(delidata_all)
delidata_all.columns = delidata_all.columns.str.lower()

# Import annotated files with convo id 
delidata_annotated = pd.read_table(f'{dir}annotated_data.tsv')
delidata_annotated.columns = delidata_annotated.columns.str.lower()


# Getting to know data soures
delidata_all['message_id'].nunique()       #36,937
delidata_annotated['message_id'].nunique() #1,696: annotated is subset of all raw data

# Generating utterances
# Extract utterances
deli_utt = delidata_all[delidata_all['message_type']=='CHAT_MESSAGE']
print(len(deli_utt)) #14,035
#%%
#Adjusting ast.literal_eval: There are some utterances covered with ""
deli_content_processed = []
for i in range(len(deli_utt)):
  if "{'message'" in deli_utt['content'].iloc[i]:
    a = ast.literal_eval(deli_utt['content'].iloc[i])['message']
    #print(a)
  elif isinstance(deli_utt['content'].iloc[i], dict):
    a = deli_utt['content'].iloc[i]['message']
  else:
    a = deli_utt['content'].iloc[i]
  deli_content_processed.append(a)
#%%
deli_utt['text'] = deli_content_processed
deli_utt = deli_utt.drop(['content'],axis=1)
#%%
#Select required columns
deli_utt = deli_utt[['message_id', 'user_name', 'user_id','timestamp','convo_id','text']].rename(columns={'user_id':'speaker','convo_id':'conversation_id'}).reset_index(drop=True)

                    
# Additional cleaning steps I think we need to do before we can convert to corpus
## Step 1: message in delidata_all['content'] column needs to be extracted in delidata_all
## Step 2: to use Corpus.from_pandas() method, we need to figure out a way to populate 
##         the reply_to field.
## Step 3: Verify that the group ID is the TSV filename. 
## Step N (? not sure if necessary for part c): Add in the metadata fields that we 
##         will need for our analysis. Number of @'s, similarity score of final answers, 
##         participant accuracy 

# STEP 2: POPULATE REPLY_TO FIELD
delidata_messages = delidata_all[delidata_all['message_type'] == "CHAT_MESSAGE"]
## a: convert timestamp to datetime 
delidata_utt = deli_utt
delidata_utt['timestamp2'] = pd.to_datetime(delidata_utt['timestamp'])
## b: sort timestamp, by convo to get correct conversation ordering
delidata_utt = delidata_utt.sort_values(['timestamp2'], ascending = True).groupby('conversation_id')
##c: create lag of message_id, rename it reply_to 
delidata_utt = delidata_utt.apply(lag, column = 'message_id').sort_values(['conversation_id', 'timestamp2'], ascending = True)
delidata_utt = delidata_utt.rename(columns={'lagged_message_id':'reply_to'}, inplace = True)
deli_utt = delidata_utt[['message_id', 'speaker', 'timestamp', 'conversation_id', 'text', 'reply_to']].rename(columns={'message_id':'id'})
deli_utt = deli_utt[['id', 'timestamp', 'text', 'speaker', 'reply_to', 'conversation_id']]




##############Corpus Construction Notes 
# Notes about a corpus: 
# Corpus is constructed of three parts: conversations, utterances, and speakers
#   (1) Utterance data frame must contain: ID, timestamp, text, speaker, 
#       reply_to, conversation_id. 
#   (2) Conversation and speaker data only contain IDs


##############Creating final answer column for meta data 
delidata_user = delidata_all[delidata_all['message_type'] == "WASON_SUBMIT"]

def parse_meta(val):
  val = eval(val)
  return ''.join([x['value'] for x in val if x['checked']])

delidata_user['meta.finalanswer']=delidata_user['content'].apply(parse_meta)
delidata_user = delidata_user.sort_values(['timestamp'], ascending = True).groupby(['convo_id', 'user_id']).tail(1)
user_df = delidata_user[['user_id', 'meta.finalanswer']]
user_df = user_df.rename(columns={'user_id': 'id'})

convo_df = delidata_all.convo_id.drop_duplicates().to_frame()
convo_df = convo_df.rename(columns={'convo_id':'id'})
# Gathering all components of the corpus


delidata_corpus = Corpus.from_pandas(utterances_df=deli_utt, speakers_df=user_df, conversations_df=convo_df)

