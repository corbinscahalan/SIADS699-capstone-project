###################
##
##  RNN tools
##
##  SIADS 699 Capstone
##
##  C. Cahalan, A. Levin-Koopman, J. Olson
##
####################


"""
Environment:

numpy=1.18.5

"""


from keras.preprocessing.text import Tokenizer

import api_utils as au
import extract_video_info as evi
import numpy as np
import pandas as pd
from typing import List, Set, Dict, Tuple, Optional
import re

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

import tensorflow as tf



def clean_subtitles(subtitles: List[str]) -> str:

    # Converts the subtitles field from a list to a single string

    # Arguments:
    # subtitles -- a list of strings

    # Returns:
    # final_subs -- a single string


    joined_subs = ' '.join(subtitles)

    final_subs = re.sub(r'\n', '', joined_subs)

    return final_subs


def prep_subtitle_data(df: pd.DataFrame, sub_field: str = 'subtitles', pop_field: str = 'pop_metric', vocab_size: int = 2500, train_size: int = 1000) -> (np.array, np.array):

    # Prepares data for RNN by cleaning and tokenizing subtitles

    # Arguments:
    # df -- Pandas dataframe of the final type 
    # sub_field -- the column name for video subtitles
    # pop_field -- the column name for the popularity metric
    # vocab_size -- the maximum number of tokens (minus one for the out of vocabulary token)
    # train_size -- token lists are truncated to this length

    # Returns:
    # sub_seqs -- List of integer lists, each one the tokenized version of the corresponding subtitle field
    # pop_vals -- List of floats, just the popularity metric pulled from df

    sub_df = df[[sub_field, pop_field]]

    sub_df[sub_field] = df[sub_field].apply(lambda x: clean_subtitles(eval(x)))

    tkn = Tokenizer(num_words=vocab_size, lower=True, split=' ', oov_token='<UNK>')

    tkn.fit_on_texts(sub_df[sub_field])

    sub_seqs = tkn.texts_to_sequences(sub_df[sub_field])

    num_rows = len(sub_seqs)

    out_array = np.zeros((num_rows, train_size))

    for i in range(num_rows):

        if len(sub_seqs[i]) >= train_size:

            out_array[i] = sub_seqs[i][:train_size]

        else:

            out_array[i] = np.pad(sub_seqs[i], (0, train_size-len(sub_seqs[i])), 'constant')


    pop_vals = np.array(sub_df[pop_field])

    return out_array, pop_vals


class rnn_master():

    # Object for managing the RNN


    def __init__(self, make_model: bool = True):

        self.vocab_size = 2500   # Number of distinct tokens to use during tokenization (minus two)
        self.train_size = 1000   # The size of the individual training instances -- number of tokens taken from each subtitle field, with padding if necessary
        self.embed_size = 100    # The dimension of the vector which token are converted to by the embedding layer
        self.num_LSTM = 32       # The number of LSTM cells in that layer
        self.num_dense = 100     # The number of nodes in the densely connected layer

        self.X = None
        self.y = None
        self.model = None

        if make_model:

            self.make_rnn()


    def prep_data(self, df: pd.DataFrame, sub_field: str = 'subtitles', pop_field: str = 'pop_metric') -> (np.array, np.array):

        # See prep_subtitle_data

        self.X, self.y = prep_subtitle_data(df, sub_field, pop_field, vocab_size=self.vocab_size, train_size=self.train_size)

        return None

    
    def make_rnn(self):

        # Instantiate a RNN model

        self.model = Sequential([
            Embedding(input_dim = self.vocab_size, input_length=self.train_size, output_dim=self.embed_size, trainable=False, mask_zero=True),
            Masking(mask_value=0.0),
            LSTM(self.num_LSTM, return_sequences=False, dropout=0.1, recurrent_dropout=0.1),
            Dense(self.num_dense, activation='relu'),
            Dense(1, activation=None)
        ])

        self.model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = 'adam', metrics = ['accuracy'])

        # TODO: Add a callback

        print(self.model.summary())



    def train_rnn(self, num_epochs: int = 5):

        # Train the RNN model with prepared data and labels

        if self.model is None:

            print("No model exists. Try make_rnn.")

            return None
        
        self.model.fit(self.X, self.y, epochs=num_epochs)



    

