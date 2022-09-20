'''
Dataset details (from rst file)
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are FOMC members, indexed by their name as recorded in the transcripts.

* id: name of the speaker
* chair: (boolean) is speaker FOMC Chair
* vice_chair: (boolean) is speaker FOMC Vice-Chair

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each utterance, we provide:

* id: index of the utterance (concatenating the meeting date with the utterance’s sequence position)
* speaker: the speaker who authored the utterance
* conversation_id: ID of meeting
* reply_to: id of the sequentially prior utterance (None for the first utterance of a meeting)
* text: textual content of the utterance
* timestamp: calculated value based off the date of the meeting and the speech index

Metadata for utterances include:

* speech_index: index of utterance in the context of the conversation
* parsed: parsed version of the utterance text, represented as a SpaCy Doc

Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversations are indexed by a string representing the meeting date.

Quick stats
-----------

For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 364
Number of Utterances: 108504
Number of Conversations: 268
'''
#%% Setup -------------------------------------------------------------------
from convokit import Corpus, download
from convokit import Transformer
import nltk
import os
import re
import pandas as pd
from tqdm import tqdm
import ast
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

wd = "/Users/jennie/Desktop/INFO6742_NLP/fomc_submission/"
outdir =  "/Users/jennie/Desktop/INFO6742_NLP/"
corpus = Corpus(f'{wd}fomc-corpus/')
#%%
df = corpus.get_utterances_dataframe()
#%%
#Calculate speakers
num_speakers = (
    df[['conversation_id','speaker']]
    .drop_duplicates()
    .groupby(['conversation_id'])
    .count()
    .reset_index()
    .rename(columns={'conversation_id':'id'}))
#%%
num_speakers.describe()
'''
         speaker
count  268.000000
mean    26.932836
std      3.939044
min     20.000000
25%     25.000000
50%     26.000000
75%     28.000000
max     61.000000
'''
#%%
for utt in corpus.iter_utterances():
    print(utt.speaker.id[:-1])
    #usernames.append(utt.speaker.id[:-1].lower())
#%%
# Now need to figure out how to str_detect all the usernames in a group and then test if the utterances in that group
conversation = corpus.random_conversation()
def prop_usernames_mentioned(convo):
    usernames = []
    for utt in convo.iter_utterances():
        usernames.append(utt.speaker.id[:-1].lower())
    print(usernames)
    j = 0
    i = 0
    for utt in convo.iter_utterances():
        i = i + 1
        target = utt.text.lower()
        target_found = []
        for user in usernames:
            target_found.append(target.find(user))
        if any([not x == -1 for x in target_found]):
            j = j + 1
    prop_utt_w_usernames = j / i

    numerator = []
    for user in usernames:
        user_found = 0
        for utt in convo.iter_utterances():
            target = utt.text.lower()
            if target.find(user) != -1:
                user_found = user_found + 1
        if user_found > 0:
            numerator.append(1)
        else:
            numerator.append(0)
    prop_users_found = sum(numerator) / len(usernames)

    convo.add_meta('prop_utt_w_usernames', prop_utt_w_usernames)
    convo.add_meta('prop_users_found', prop_users_found)
    return None


class NewSignal(Transformer):
    def transform(self, corpus):
        for convo in corpus.iter_conversations():
            prop_usernames_mentioned(convo)
        return (corpus)

#%%
ns = NewSignal()
corpus = ns.transform(corpus)
#%%
convo_df = (
    corpus
    .get_conversations_dataframe()
    .drop('vectors', axis = 1)
    .reset_index()
    )
#%%
convo_df
#%%
convo_df = pd.merge(convo_df, num_speakers)
#%%
convo_df = convo_df.astype(float)
#%%

sns.set(font_scale=0.8)
#sns.histplot(num_speakers)
a = sns.histplot(convo_df['meta.prop_utt_w_usernames']).set(title='Frequency of Proportion Utterance w/ Usernames')
#a.figure.tight_layout()
plt.show()
#%%
convo_df['meta.prop_utt_w_usernames'].describe()
#%%
sns.set(font_scale=0.8)
#sns.histplot(num_speakers)
a = sns.histplot(convo_df['meta.prop_users_found']).set(title='Frequency of Proportion Users Found')
#a.figure.tight_layout()
plt.show()
#%%
convo_df['meta.prop_users_found'].describe()
#%%
sns.pairplot(convo_df, kind="reg", plot_kws={'line_kws':{'color':'red'}})
plt.show()
#%%
sns.set(font_scale=0.8)
heat_convo = convo_df.corr()
ht_con = sns.heatmap(heat_convo, annot=True)
ht_con.figure.tight_layout()
plt.show()
#%%