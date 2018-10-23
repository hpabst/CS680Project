from metrics import *
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import random


def evaluate_model(model, X_train, y_train, X_val, y_val):
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    prec_train = precision(y_train, pred_train)
    prec_val = precision(y_val, pred_val)
    recall_train = recall(y_train, pred_train)
    recall_val = recall(y_val, pred_val)
    f1_train = fbeta_score(y_train, pred_train)
    f1_val = fbeta_score(y_val, pred_val)
    val_results = "Validation Results: Prec - {:.2f}  Recall - {:.2f}  F1 - {:.2f}".format(prec_val,
                                                                                        recall_val,
                                                                                        f1_val)
    train_results = "Train Results:      Prec - {:.2f}  Recall - {:.2f}  F1 - {:.2f}".format(prec_train,
                                                                                          recall_train,
                                                                                          f1_train)
    print("{}\n{}".format(train_results, val_results))
    return

def shuffle_arrays(arr1, arr2, seed=None):
    if seed is None:
        seed = random.randint()
    indices = np.arange(arr1.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)
    arr1_ret = arr1[indices]
    arr2_ret = arr2[indices]
    return arr1_ret, arr2_ret


def create_embedding_matrix(filepath, dictionary, embedding_dim):
    embeddings_index = {}
    with open(filepath, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(dictionary)+1, embedding_dim))
    for word, i in dictionary.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def read_data_embeddings():
    kdd_data = pd.read_csv("../data/KDD_Cup_2005_Data.csv")
    kdd_x = kdd_data[["Query"]]
    kdd_y = kdd_data[["Label"]]
    google_trends_data = pd.read_csv("../data/Google_Trends_Search_Queries.csv")
    trends_x = google_trends_data[["Query"]]
    trends_y = google_trends_data[["Label"]]
    X = pd.concat((kdd_x, trends_x))
    y = pd.concat((kdd_y, trends_y))
    tokenizer = Tokenizer(num_words=200000, split=' ')
    tokenizer.fit_on_texts(X["Query"].values)
    X_train = tokenizer.texts_to_sequences(X["Query"].values)
    X_train = pad_sequences(X_train)
    y_train = y["Label"].values
    return X_train, y_train