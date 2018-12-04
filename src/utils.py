from metrics import *
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import random
import pickle
import json
import pathlib

MAX_NUM_WORDS = 200000


def evaluate_model(model, X_train, y_train, X_val, y_val):
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    prec_train = precision(y_train, pred_train)
    prec_val = precision(y_val, pred_val)
    recall_train = recall(y_train, pred_train)
    recall_val = recall(y_val, pred_val)
    f1_train = fbeta_score(y_train, pred_train)
    f1_val = fbeta_score(y_val, pred_val)
    auc = roc_auc_score(y_val, pred_val)
    val_results = "Validation Results: Prec - {:.2f}  Recall - {:.2f}  F1 - {:.2f}  AUC - {:.2f}".format(prec_val,
                                                                                                recall_val,
                                                                                                 f1_val,
                                                                                                auc)
    train_results = "Train Results:      Prec - {:.2f}  Recall - {:.2f}  F1 - {:.2f}".format(prec_train,
                                                                                          recall_train,
                                                                                          f1_train)
    print("{}\n{}".format(train_results, val_results))
    return prec_val, recall_val, f1_val


def save_model(model, history, filename,
               prec=None, recall=None, f1=None,
               epochs=None, batch_size=None, random_seed=None,
               class_weights=None, sample_type=None):
    pathlib.Path("../models/{}".format(filename)).mkdir(parents=True, exist_ok=True)
    model.save("../models/{}/{}.h5".format(filename, filename))
    with open("../models/{}/{}.txt".format(filename, filename), 'wb') as f:
        pickle.dump(history.history, f)
    with open("../results/{}.txt".format(filename), 'w+') as f:
        if prec is not None:
            f.write("Validation Set Precision: {}\n".format(prec))
        if recall is not None:
            f.write("Validation Set Recall   : {}\n".format(recall))
        if f1 is not None:
            f.write("Validation Set F1 Score : {}\n".format(f1))
        if epochs is not None:
            f.write("Number epochs trained: {}\n".format(epochs))
        if batch_size is not None:
            f.write("Batch size: {}\n".format(batch_size))
        f.write("Random seed: {}\n".format(random_seed))
        if class_weights is not None:
            f.write(json.dumps(class_weights))
            f.write("\n")
        if sample_type is not None:
            f.write("Sample type: {}\n".format(sample_type))
    plot_model(model, to_file="../models/{}/{}.png".format(filename, filename), show_shapes=True)
    return


def shuffle_arrays(arr1, arr2, seed=None):
    if seed is None:
        seed = random.randint(0, 1000)
    indices = np.arange(arr1.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)
    arr1_ret = arr1[indices]
    arr2_ret = arr2[indices]
    return arr1_ret, arr2_ret


def is_digit(string):
    #  Sourced from https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
    try:
        float(string)
        return True
    except ValueError:
        return False


def collapse_embedding_array(array):
    word = array[0]
    remaining_arr = []
    for i in range(1, len(array)):
        if not is_digit(array[i]):
            word += (" " + array[i])
        else:
            remaining_arr = array[i:]
            break
    return [word] + remaining_arr


def create_embedding_matrix(filepath, dictionary, embedding_dim):
    embeddings_index = {}
    with open(filepath, encoding="utf8") as f:
        for line in f:
            values = collapse_embedding_array(line.split())
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(dictionary)+1, embedding_dim))
    for word, i in dictionary.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def recover_queries(tokenizer, X, y):
    recovered_queries = []
    for i, query in enumerate(X):
        query_text = []
        for index in query:
            query_text.append(tokenizer.index_word.get(index, "UNKNOWN"))
        if y[i] == 1:
            recovered_queries.append(query_text)
    return recovered_queries


def read_data_embeddings(max_input_length=10, vocab_size=MAX_NUM_WORDS):
    X_train, y_train, X_val, y_val = read_train_val_data()
    X = pd.concat((X_train, X_val))
    tokenizer = Tokenizer(num_words=vocab_size, split=' ')
    tokenizer.fit_on_texts(X[0].values)
    X_train = tokenizer.texts_to_sequences(X_train[0].values)
    X_train = pad_sequences(X_train, maxlen=max_input_length)
    X_val = tokenizer.texts_to_sequences(X_val[0].values)
    X_val = pad_sequences(X_val, maxlen=max_input_length)
    y_train = y_train[0].values
    y_val = y_val[0].values
    return X_train, y_train, X_val, y_val, tokenizer


def read_bag_of_words(vocab_size=MAX_NUM_WORDS):
    X_train, y_train, X_val, y_val = read_train_val_data()
    X_total = pd.concat((X_train, X_val))
    tokenizer = Tokenizer(num_words=vocab_size, split=' ')
    tokenizer.fit_on_texts(X_total[0].values)
    X_train = tokenizer.texts_to_matrix(X_train[0].values)
    X_val = tokenizer.texts_to_matrix(X_val[0].values)
    y_train = y_train[0].values
    y_val = y_val[0].values
    return X_train, y_train, X_val, y_val, tokenizer


def validation_training_split(split=0.2):
    kdd_data = pd.read_csv("../data/KDD_Cup_2005_Data.csv")
    kdd_x = kdd_data[["Query"]]
    kdd_y = kdd_data[["Label"]]
    google_trends_data = pd.read_csv("../data/Google_Trends_Search_Queries.csv")
    trends_x = google_trends_data[["Query"]]
    trends_y = google_trends_data[["Label"]]
    kdd_data_8k = pd.read_csv("../data/KDD_Cup_2005_Data_8k.csv")
    kdd_8k_x = kdd_data_8k[["Query"]]
    kdd_8k_y = kdd_data_8k[["Label"]]
    X = pd.concat((kdd_x, kdd_8k_x, trends_x))
    y = pd.concat((kdd_y, kdd_8k_y, trends_y))
    X, y = shuffle_arrays(X.values, y.values)
    num_val_examples = int(split * X.shape[0])
    x_train = X[:-num_val_examples]
    y_train = y[:-num_val_examples]
    x_val = X[-num_val_examples:]
    y_val = y[-num_val_examples:]
    pd.DataFrame(x_train).to_csv("../data/X_train.csv", header=None, index_label=False, index=False)
    pd.DataFrame(y_train).to_csv("../data/y_train.csv", header=None, index_label=False, index=False)
    pd.DataFrame(x_val).to_csv("../data/X_val.csv", header=None, index_label=False, index=False)
    pd.DataFrame(y_val).to_csv("../data/y_val.csv", header=None, index_label=False, index=False)
    return


def read_train_val_data():
    x_train = pd.read_csv("../data/X_train.csv", header=None)
    y_train = pd.read_csv("../data/y_train.csv", header=None)
    x_val = pd.read_csv("../data/X_val.csv", header=None)
    y_val = pd.read_csv("../data/y_val.csv", header=None)
    return x_train, y_train, x_val, y_val
