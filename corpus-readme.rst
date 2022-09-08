DeliData Corpus
============================
Chatroom transcripts from experiments (in-lab and MTurk) where participants engage in a cognitive task and collectively decide on what they think is the correct answer for the Wason card game. In the Wason card game, participants are shown 4 cards and are asked to select the cards that they would need to test the following rule: "All cards with vowels on one side, have an even number on the other." There are 500 conversations with an average of 28 utterances each. Each conversation involves 2-5 participants discussing their card selection choices. 

Paper: Kharadzhov, G., Stafford, T., & Vlachos, A. (2021). DeliData: A dataset for deliberation in multi-parthy problem solving. arXiv prepint arXiv:2108.05271. 


Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are participants in a group of 2-5 people for the Wason card selection game.  
* id: the unique identifier for each speaker  

Metadata for speakers: 
* final_answer: the final card selection of each speaker. Found by identifying the latest Wason submission for each user by group. Example values: ‘5E’, ‘3A’, ‘69EB’. 
* correct_answer: a flag equal to 1 if the speaker's final card selection was correct (i.e. the speaker selected a vowel AND an odd number) and equal to 0 if the speaker's final card selection was incorrect. 


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each chat message is viewed as an utterance. For each utterance, we provide:

* id: index of the utterance
* speaker: the speaker who authored the utterance. 
* conversation_id: id of the first utterance in the conversation this utterance belongs to
* reply_to: id of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance


Metadata for utterances: N/A



Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each chat room is considered a conversation. For each conversation, we provide the following metadata:

* num_participants: The number of participants in a chatroom. For a group of two, num_participants = 2. For a group of five, num_participants = 5. 
* num_chats: The total number of chats that occurred in the chatroom. 
* agreement_score: A ratio that measures the total number of answers that are the same divided by the number of group members. For example, if there were 5 people in a group and 2 people submitted the same card selections, the ratio would be 2/5 or 0.4. 
* correct_score: A ratio that measures the total number of final card submissions that were correct (i.e. when card selection includes a vowel AND an odd number).For example, if there were 5 people in a group and 4 people submitted the correct card selection, the ratio would be 4/5 or 0.80. 




Corpus-level information
^^^^^^^^^^^^^^^^^^^^^^^^

There is no additional information for this corpus.

Usage
-----

Extract files to a local directory.

>>> corpus = Corpus(filename='corpus/directory/path')

For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 8096
Number of Utterances: 61206
Number of Conversations: 6682

Additional note
---------------

The original dataset can be downloaded `here <https://www.delibot.xyz/delidata/>`_.

Contact
^^^^^^^

Please email any questions to: jl3369@cornell.edu (Jinsook Lee) or sg972@cornell.edu (Sammy Gold).
