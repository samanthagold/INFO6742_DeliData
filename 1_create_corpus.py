# GOAL: transform corpus into ConvoKit format 
## Made a few assumptions: 
  # - we want to use all raw files from delidata (not just the Mturk files
  #   or just the pilot files)

# Allows me to use Python in RStduio
reticulate::repl_python()

# Packages 
import convokit 
from convokit import Corpus, download 
import nltk
import os
import pandas as pd

# Setup 
dir = "Downloads/delidata/"

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




  
