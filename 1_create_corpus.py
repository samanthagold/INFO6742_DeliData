# GOAL: transform corpus into ConvoKit format 
## Made a few assumptions: 
  # - we want to use all raw files from delidata (not just the Mturk files
  #   or just the pilot files)
import ast

# Packages 
from convokit import Corpus, download
from convokit import Transformer
import nltk
import os
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

# Setup 
dir = "/Users/jennie/PycharmProjects/INFO6742_DeliData/data/delidata/" #Please change it to your directory
#%%
############### Data Prep 
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
# Import annotated files with convo id 
delidata_annotated = pd.read_table(f'{dir}annotated_data.tsv')
delidata_annotated.columns = delidata_annotated.columns.str.lower()

# Getting to know data sources
delidata_all['message_id'].nunique()       #36,937
delidata_annotated['message_id'].nunique() #1,696: annotated is subset of all raw data
#%%
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
#%%
#Save processed utterance dataframe
deli_utt.to_csv(dir+"deli_utt.csv", sep="|")
#%%
########GOT STUCK HERE########## We need 'reply_to' column...
new_corpus = Corpus.from_pandas(deli_utt)

#%%




class ShoutingTransformer(Transformer):
  def __init__(self):
    self.meta_name = 'is_shouting'

  def transform(self, corpus):
    for utt in corpus.iter_utterances():
      utt.meta[self.meta_name] = utt.text.isupper()
    return corpus




# Additional cleaning steps I think we need to do before we can convert to corpus
## Step 1: message in delidata_all['content'] column needs to be extracted in delidata_all
## Step 2: to use Corpus.from_pandas() method, we need to figure out a way to populate 
##         the reply_to field.
## Step 3: Verify that the group ID is the TSV filename. 
## Step N (? not sure if necessary for part c): Add in the metadata fields that we 
##         will need for our analysis. Number of @'s, similarity score of final answers, 
##         participant accuracy 


##############Corpus Construction Notes 
# Notes about a corpus: 
# Corpus is constructed of three parts: conversations, utterances, and speakers
#   (1) Utterance data frame must contain: ID, timestamp, text, speaker, 
#       reply_to, conversation_id. 
#   (2) Conversation and speaker data only contain IDs




  
