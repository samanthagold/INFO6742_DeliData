###Extract simple statistics (e.g., number of conversations involving at least k tall people).
# Post afollow-up on your A1B Piazza post discussing the statistics that are most relevant to your hypotheses
# (as well as any inconsistencies you might discover)

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
#%%
#load the corpus
corpus = Corpus("/Users/jennie/Desktop/INFO6742_NLP/Group_Project/delidata_corpus")
#%%
#Simple Stat
df = (
    corpus
      .get_utterances_dataframe()[['conversation_id','timestamp','speaker','text']]
      .reset_index()
)
#%%
#create a column if each utterance contains "@"
df['@_yn'] = ""

for i in range(len(df)):
    if "@" in df['text'].iloc[i]:
        df['@_yn'].iloc[i] = 1
        print(df['conversation_id'].iloc[i])
        print(df['text'].iloc[i])
    else:
        df['@_yn'].iloc[i] = 0
#%%
#################################
###########SIMPLE STAT###########
#################################
#Simple Stat 0 - length of each convo

length_convo = (
    df
    .groupby(['conversation_id'])
    .count()
    .reset_index()[['conversation_id','id']]
    .rename(columns={'id':'len_convo'})
)

print(length_convo.len_convo.describe())
'''
mean      28.042000
std       15.136792
min        7.000000
25%       18.000000
50%       25.000000
75%       34.000000
max      104.000000
'''
##Histogram
sns.histplot(length_convo).set(title='Distribution of Conversation Length')
plt.show()
#%%
#Simple Stat 1 - number of "@" in each convo
number_at = (
    df
    .groupby(['conversation_id'])
    .sum()
    .reset_index()[['conversation_id','@_yn']]
    .rename(columns={'@_yn':'num_at'})
    )

print(len(number_at[number_at['num_at']>0])) #count no. of convo that has "@" at least once; 101
print(len(number_at[number_at['num_at']==0])) #count no. of convo that has no "@"; 399
print(number_at[number_at['num_at']>0].describe())
'''
count  101.000000
mean     2.287129
std      1.756910
min      1.000000
25%      1.000000
50%      2.000000
75%      3.000000
max     11.000000
'''
##Histogram
sns.histplot(data=number_at[number_at['num_at']>0]).set(title = 'Distribution of "@"') #convo w/o @ excluded
plt.show()
#%%
#Simple Stat 2 - proportion of "@" from each convo (normalized)
convo_stat = pd.merge(number_at, length_convo)
convo_stat['prop_at'] = convo_stat['num_at']/convo_stat['len_convo']

print(convo_stat['prop_at'].describe())
'''
count    500.000000
mean       0.015158
std        0.038325
min        0.000000
25%        0.000000
50%        0.000000
75%        0.000000
max        0.323529
Name: prop_at, dtype: float64
'''
#%%
test = convo_stat[convo_stat['num_at']!=0]
print(test['prop_at'].describe())
'''
count    101.000000
mean       0.075040
std        0.052829
min        0.011364
25%        0.038462
50%        0.060606
75%        0.100000
max        0.323529
'''

#sns.histplot(convo_stat['prop_at']).set_title("Frequency of @ proportion (convo with zero @ included)")
sns.histplot(test['prop_at']).set_title("Frequency of @ proportion (convo with zero @ excluded)")
plt.show()

#%%
#Simple Stat 3 - first "@" appearance timing (e.g. if an utterance with "@" appeared 3rd out of 10 utterances it's 0.3, smaller means earlier)
convo_list = df['conversation_id'].unique().tolist()
#%%
timing = []
for i in convo_list:
    for j in range(len(df[df['conversation_id'].isin([i])])):
        if "@" in df[df['conversation_id'].isin([i])]['text'].iloc[j]:
            timing.append([i,j])

timing = pd.DataFrame(timing, columns=['conversation_id','order_at'])
timing = (
    timing
    .groupby('conversation_id')['order_at']
    .apply(lambda grp: grp.nsmallest(1))
    .reset_index()[['conversation_id','order_at']]
)

#%%
##Merging with convo_stat
convo_stat = pd.merge(convo_stat, timing, how='left')
convo_stat['first_at_timing'] = convo_stat['order_at']/convo_stat['len_convo']
#%%
#convo_stat['first_at_timing'] =convo_stat['first_at_timing'].fillna(1.1) #convo with "@" filled with 1.1
#%%
print(convo_stat['first_at_timing'].describe())
#%%
##Histogram
sns.histplot(convo_stat['first_at_timing']).set(title = 'Frequency of first appearance timing of @')
plt.show()
#%%
#Simple Stat 4 - Interval?... no not enough data so no point to calculate interval of @
#%%
#Simple Stat 5 - Agreeableness
agree = []
for i in df['speaker'].unique().tolist():
    speaker = corpus.get_speaker(i)
    final_answer = speaker.meta['finalanswer']
    #correct_flag = speakermeta['correct_flag']
    agree.append([i, final_answer])

agree = pd.DataFrame(agree, columns=['speaker','finalanswer'])
agree = pd.merge(df[['conversation_id','speaker']].drop_duplicates(), agree, how='left')
#%%
agree_ent = agree.groupby(['conversation_id','finalanswer']).count().reset_index()
agree['num_speakers'] = 1
num_sp = agree.groupby(['conversation_id']).sum().reset_index()
agree_ent = pd.merge(agree_ent, num_sp)
agree_ent['answer_ratio'] = agree_ent['speaker']/agree_ent['num_speakers']
#%%
entr = []
for i in convo_list:
    for j in range(len(agree_ent[agree_ent['conversation_id'].isin([i])])):
        en = agree_ent[agree_ent['conversation_id'].isin([i])]
        print(en)
        ent = stats.entropy(en['answer_ratio'].tolist(), base=2)
        entr.append(ent)
#%%
agree_ent['agreeableness'] = entr
agree_ent = agree_ent[['conversation_id','agreeableness']].fillna(0).drop_duplicates().reset_index(drop=True)
#%%
convo_stat=pd.merge(convo_stat,agree_ent)
convo_stat['agreeableness'] = 1-convo_stat['agreeableness'] #to make it more intuitive
#%%
sns.histplot(convo_stat['agreeableness']).set(title='Frequency of Agreeableness')
plt.show()
#%%
print(convo_stat['agreeableness'].describe())
#%%
#Simple Stat 6 - proportion of accuracy (e.g. if 3 people got correct out of 5 it's 0.6)
# --> it's in convo meta data
correct = []
for i in convo_list:
    con = corpus.get_conversation(i)
    correct_ratio = con.meta['correct_ratio']
    #correct_flag = speakermeta['correct_flag']
    correct.append([i, correct_ratio])

correct = pd.DataFrame(correct, columns=['conversation_id','correct_ratio'])
convo_stat = pd.merge(correct, convo_stat)
#%%
sns.histplot(convo_stat['correct_ratio']).set(title='Frequency of Correct Ratio')
plt.show()
#%%
print(convo_stat['correct_ratio'].describe())
#%%
#################################
###########COREELATION###########
#################################
sns.set(font_scale=2)
sns.pairplot(convo_stat, kind="reg", plot_kws={'line_kws':{'color':'red'}})
plt.show()
#%%
test = convo_stat[~convo_stat['first_at_timing'].isnull()]
sns.pairplot(test, kind="reg", plot_kws={'line_kws':{'color':'red'}})
plt.show()
#%%
sns.set(font_scale=0.8)
heat_convo = convo_stat.corr()
ht_con = sns.heatmap(heat_convo, annot=True)
ht_con.figure.tight_layout()
plt.show()
#%%
heat_convo_test = test.corr()
ht_con_t = sns.heatmap(heat_convo_test, annot=True)
ht_con_t.figure.tight_layout()
plt.show()
#Question: should we consider the utterances w/o "@" but calling someone's name as same as "@"?
