# Goal: Use corpus to explore hypothesis that more '@'s = more conversation = more accuracy 



# Setup -------------------------------------------------------------------
from convokit import Corpus, download, TextParser, Transformer, PolitenessStrategies, Classifier
import nltk
import matplotlib as mpl

wd = "/Users/sammygold/Desktop/INFO6742/INFO6742_DeliData/"
outdir =  "/Users/sammygold/Desktop/INFO6742/INFO6742_DeliData/part_d/"
delidata_corpus = Corpus(f'{wd}delidata_corpus/')


# PART A: Summary Stats -----------------------------------------------------------




# PART B: Use politeness transformer  -------------------------------------
parser = TextParser(verbosity = 1000)
delidata_corpus = parser.transform(delidata_corpus)
ps = PolitenessStrategies()
delidata_corpus = ps.transform(delidata_corpus, markers=True)
delidata_corpus.dump(f'{wd}part_d/annotated_corpus/')
# Graphing the politeness features 
fig1 = ps.summarize(delidata_corpus, plot = True) 
# Cool, we've got politeness strategies! 

# Now looking at how politeness strategies vary with our meta data
# First, we need to get the politeness features into a nice dataset at
# the convo-level 

# From Daniel - conversation level strategy worked on together 9/11
# re-write 
def extract_strategies(convo, strategies):
    convo_strategies = pd.DataFrame(strategies.summarize(convo))
    convo_strategies = (
        convo_strategies
        .reset_index()
        .rename(columns={'index':'strategy', 0:'proportion'})
    )
    convo_strategies.index = [convo.id]*len(convo_strategies)
    convo_strategies['strategy'] = convo_strategies['strategy'].apply(
        lambda x: x.removeprefix('feature_politeness_==').removesuffix('==')
    )
    convo_strategies = convo_strategies.pivot(
        index=None,
        columns='strategy',
        values='proportion'
    )
    convo_strategies = (
        convo_strategies
        .reset_index()
        .rename(columns={'index':'id'})
    )
    return convo_strategies

strategies_df = []
for convo in delidata_corpus.iter_conversations():
    strategies = extract_strategies(convo, ps)
    strategies_df.append(strategies)
strategies_df = pd.concat(strategies_df)

# Merging onto convo_df
convo_df = (
    delidata_corpus
    .get_conversations_dataframe()
    .drop('vectors', axis = 1)
    .reset_index() 
    .merge(strategies_df, how = 'inner', on = 'id')
)

# Writing out df 
convo_df.to_csv(f'{outdir}convo_w_polite.csv')

# Looking at scatter plots between key convo meta data fields 
fig2 = convo_df.plot.scatter(x='meta.num_chats', y='1st_person_start')
    




