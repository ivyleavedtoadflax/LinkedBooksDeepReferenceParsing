import logging
import os
import random
from code.models import BiLSTM_predict
from code.utils import (characterLevelData, characterLevelIndex,
                        encodePadData_x, encodePadData_y, indexData_x,
                        indexData_y, load_data, mergeDigits, read_json)

import numpy as np
import tensorflow
import ipdb

# Seed
random.seed(42)
np.random.seed(42)
tensorflow.compat.v1.set_random_seed(42)


logger = logging.getLogger(__name__)


# Load previously cached objects created by # keras/main_threeTasks.py

inter_path = os.path.join("keras", "data")

char2ind = read_json(os.path.join(inter_path, "char2ind.json"))
ind2word = read_json(os.path.join(inter_path, "ind2word.json"), keys_to_int=True)
word2ind = read_json(os.path.join(inter_path, "word2ind.json"))
label2ind3 = read_json(os.path.join(inter_path, "label2ind3.json"))
ind2label3 = read_json(os.path.join(inter_path, "ind2label3.json"), keys_to_int=True)
maxes = read_json(os.path.join(inter_path, "maxes.json"))

maxChar = maxes["maxChar"]
maxWords = maxes["maxWords"]
maxlen = maxes["maxlen"]

# Load entire data

logger.info("Loading data")

# Load just word data (no tags)

ipdb.set_trace()

# X_test_w: list of lists where each list is a line 
# (which may contain a reference)

X_test_w, y_test1_w, y_test2_w, y_test3_w = load_data("dataset/clean_test.txt")


digits_word = "$NUM$"

X_test_w = mergeDigits([X_test_w], digits_word)

# Subset the list (it only contains one thing)
# This may not be right(!?)

X_test_w = X_test_w[0]

# Compute indexes for words+labels in the training data
# Out-of-vocabulary words entry in the "words to index" dictionary

# Convert data into indexes data

# 'pre' or 'post': Style of the padding, in order to have sequence
# of the same size.

padding_style = 'pre'

# Out-of-vocabulary words entry in the "words to index" dictionary

ukn_words = "out-of-vocabulary"

# X_test.shape = (2258, 54)
# Vector of length 54 for each sample in X_test_w

X_test = encodePadData_x(
    x=X_test_w,
    word2ind=word2ind,
    maxlen=maxlen,
    ukn_words=ukn_words,
    padding_style=padding_style
)

# X_test_char.shape = (2258, 73, 54)
# Each character has a vector of length ? within each token vector

ipdb.set_trace()

X_test_char = characterLevelData(
    X=X_test_w,
    char2ind=char2ind,
    maxWords=maxWords,
    maxChar=maxChar,
    digits_word=digits_word,
    padding_style=padding_style
)

# Training, Tesing and Validation data for the model (word emb + char features)

X_testing = [X_test, X_test_char]


out = BiLSTM_predict(
    data=X_testing,
    output="crf",
    model_weights="model_results/old/task3/task3.h5",
    word2ind=word2ind,
    maxWords=maxWords,
    ind2label=[ind2label3],
    pretrained_embedding=True,
    word_embedding_size=300,
    maxChar=maxChar,
    char_embedding_type="BILSTM",
    char2ind=char2ind,
    char_embedding_size=100,
    lstm_hidden=200
)

print(out)
