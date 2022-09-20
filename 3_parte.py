# Goal: Use corpus to explore hypothesis that more '@'s = more conversation = more accuracy 



# Setup -------------------------------------------------------------------
from convokit import Corpus, download, TextParser, Transformer, PolitenessStrategies, Classifier
import nltk
import matplotlib as mpl

wd = "/Users/sammygold/Desktop/INFO6742/INFO6742_DeliData/"
outdir =  "/Users/sammygold/Desktop/INFO6742/INFO6742_DeliData/part_e/"
delidata_corpus = Corpus(f'{wd}delidata_corpus/')


# Coding up username analysis -------------------------------------------------------------------
## first converting to dataframes to work with it easier 

convo_df = (
    delidata_corpus
    .get_conversations_dataframe()
    .drop('vectors', axis = 1)
    .reset_index()
    .rename(columns = {'id': 'conversation_id'})
    
    
)

speaker_df = (
    delidata_corpus
    .get_speakers_dataframe()
    .drop('vectors', axis = 1) 
    .reset_index()
    .rename(columns = {'id': 'speaker_id'})
    
)

utterances_df = (
    delidata_corpus
    .get_utterances_dataframe()
    .drop('vectors', axis = 1)
    .reset_index()
    .merge(convo_df, how = "inner", on = "conversation_id")
    .merge(speaker_df, how = "inner", right_on = 'speaker_id', left_on = 'speaker')
    .sort_values(['timestamp', 'conversation_id'], ascending = True)
)

# Now need to figure out how to str_detect all the userames in a group and then test if the utterances in that group 

conversation = delidata_corpus.random_conversation()
def prop_usernames_mentioned(convo):
    usernames = []
    for utt in convo.iter_utterances():
        usernames.append(utt.speaker.meta['username'].lower())
    usernames = list(set(usernames))
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
    prop_utt_w_usernames = j/i
    
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
    prop_users_found = sum(numerator)/len(usernames)
            
    convo.add_meta('prop_utt_w_usernames', prop_utt_w_usernames)
    convo.add_meta('prop_users_found', prop_users_found)
    return None

class NewSignal(Transformer):
    def transform(self, corpus):
        for convo in corpus.iter_conversations(): 
            prop_usernames_mentioned(convo)
        return(corpus)

ns = NewSignal()
delidata_corpus = ns.transform(delidata_corpus)


convo_df = (
    delidata_corpus
    .get_conversations_dataframe()
    .drop('vectors', axis = 1)
    .reset_index()
    )


convo_df.to_csv(f'{outdir}convo_w_polite.csv', index = False)








