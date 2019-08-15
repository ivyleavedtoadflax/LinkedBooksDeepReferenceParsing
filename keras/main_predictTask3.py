import logging
import os
import random
from code.models import BiLSTM_predict
from code.utils import (characterLevelData, characterLevelIndex,
                        encodePadData_x, encodePadData_y, indexData_x,
                        indexData_y, load_data, mergeDigits, read_json)
import json
import numpy as np
import pandas as pd
import tensorflow

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


# X_test_w: list of lists where each list is a line
# (which may contain a reference)

X_test_w, y_test1_w, y_test2_w, y_test3_w = load_data("dataset/clean_test.txt")

with open("/home/matthew/Documents/wellcome/datalabs/modelling/policy/reference_labelling/l10_conll_format.txt") as fb:
    out = fb.read()

import ipdb
ipdb.set_trace()

X_test_w = out



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

# X_test.shape = (2258, 73)
# Vector of length 73 for each sample in X_test_w

X_test = encodePadData_x(
    x=X_test_w,
    word2ind=word2ind,
    maxlen=maxlen,
    ukn_words=ukn_words,
    padding_style=padding_style
)

# X_test_char.shape = (2258, 73, 54)
# Each character has a vector of length ? within each token vector

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


def create_prodigy_format(tokens, labels):

    prodigy_data = []

    all_token_index = 0

    for line in tokens:
        prodigy_example = {}
        line_token_index = 0

        tokens = []
        spans = []
        token_start = 0

        for i, token in enumerate(line):

            token_end = token_start + len(token)

            tokens.append({
                "text": token, 
                "id": line_token_index,
                "start": token_start,
                "end": token_end,
            })


            spans.append({
                "label": labels[line_token_index],
                "start": token_start,
                "end": token_end,
                "token_start": line_token_index,
                "token_end": line_token_index,
                "text": token,
            })

            prodigy_example["text"] = " ".join(line)
            prodigy_example["tokens"] = tokens
            prodigy_example["spans"] = spans
            prodigy_example["meta"] = {"line": i}

            line_token_index += 1
            all_token_index += 1
            token_start += token_end + 1

        prodigy_data.append(prodigy_example)

    return prodigy_data

def write_jsonl(input_data, outfile, shuffle=False):
    """Write a dict or list to a newline delimited json lines file

    Sentences are shuffled first if shuffle=True is passed
    """
    with open(outfile, "w") as fb:

        # Check if a dict (and convert to list if so)

        if isinstance(input_data, dict):
            input_data = [value for key, value in input_data.items()]

        # If shuffle=True shuffle the data

        if shuffle:
            input_data = random.sample(input_data, k=len(input_data))

        # Write out to jsonl file

        logger.info("Writing jsonl file to file: %s", outfile)

        for i in input_data:
            json_ = json.dumps(i) + "\n"
            fb.write(json_)

flat_x = [item for sublist in X_test_w for item in sublist]
foo = list(zip(flat_x, out))
df = pd.DataFrame(foo, columns=["token", "label"])
df.to_csv("prediction.csv", index=False)

prod_format = create_prodigy_format(tokens=X_test_w, labels=out)

write_jsonl(prod_format, "predictions.jsonl")




