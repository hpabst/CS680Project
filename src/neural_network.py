from utils import *
from data_rebalancing import balance_samples
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, Dropout, Embedding, Flatten
from keras.models import Model, Sequential, load_model
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os

VAL_SPLIT = 0.2
MAX_QUERY_LENGTH = 10
VOCAB_SIZE = 300
EMBEDDING_DIM = 300
EMBEDDING_DIR = "../embeddings/glove.6B.300d.txt"

SEED = None
EPOCHS = 50
BATCH_SIZE = 64
WEIGHT_CLASSES = False
SAMPLE_TYPE = None
PRETRAIN_EMBEDDINGS = True


def evaluate_all_models():
    root_dir = "../models"
    X_train_de, y_train_de, X_val_de, y_val_de, tokenizer = read_data_embeddings(max_input_length=MAX_QUERY_LENGTH)
    X_train_bow, y_train_bow, X_val_bow, y_val_bow, _ = read_bag_of_words(vocab_size=VOCAB_SIZE)
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if ".h5" in file and "weightclasses" not in file:
                print("Model name: {}".format(file))
                model = load_model(os.path.join(subdir, file))
                if "embed" in file:
                    evaluate_model(model, X_train_de, y_train_de, X_val_de, y_val_de)
                elif "bow5000" in file:
                    evaluate_model(model, X_train_bow, y_train_bow, X_val_bow, y_val_bow)
                print("")
    return


def nn_bag_of_words():
    print("Creating bags of words...")
    X_train, y_train, X_val, y_val, tokenizer = read_bag_of_words(vocab_size=VOCAB_SIZE)
    print("X_train shape: {}".format(X_train.shape))
    print("y_train shape: {}".format(y_train.shape))

    print("X_val shape: {}".format(X_val.shape))
    print("y_val shape: {}".format(y_val.shape))

    train_positive = np.sum(y_train)
    num_train = y_train.shape[0]

    if WEIGHT_CLASSES:
        class_weights = {
            0: 1.,
            1: (num_train-train_positive)/train_positive
        }
    else:
        class_weights = {
            0: 1.,
            1: 1.,
        }

    print("Rebalancing data...")
    X_train, y_train = balance_samples(X_train, y_train, seed=SEED, type=SAMPLE_TYPE)

    print("Generating NN model...")
    model = Sequential()
    model.add(Dense(2048, input_shape=(VOCAB_SIZE,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    early_stopper = EarlyStopping(patience=5)
    optimizer = Adam(lr=0.0001)
    print("Compiling model...")
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print("Training model...")
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(X_val, y_val), verbose=2,
                        class_weight=class_weights,
                        callbacks=[early_stopper])

    print("Determining validation metrics...")
    val_prec, val_recall, val_f1 = evaluate_model(model, X_train, y_train, X_val, y_val)
    model_name = input("Enter a filename for model >> ")
    save_model(model, history, model_name,
               prec=val_prec, recall=val_recall, f1=val_f1,
               epochs=EPOCHS, batch_size=BATCH_SIZE, random_seed=SEED,
               class_weights=class_weights, sample_type=SAMPLE_TYPE)
    return model


def nn_word_embeddings():
    x_train, y_train, x_val, y_val, tokenizer = read_data_embeddings(max_input_length=MAX_QUERY_LENGTH)
    train_positive = np.sum(y_train)
    num_train = y_train.shape[0]

    if WEIGHT_CLASSES:
        class_weights = {
            0: 1.,
            1: (num_train-train_positive)/train_positive,
        }
    else:
        class_weights = {
            0: 1.,
            1: 1.,
        }

    print("Rebalancing training data...")
    x_train, y_train = balance_samples(x_train, y_train, seed=SEED, type=SAMPLE_TYPE)

    print("Creating word embeddings matrix...")
    embedding_matrix = create_embedding_matrix(EMBEDDING_DIR,
                                               tokenizer.word_index,
                                               EMBEDDING_DIM)

    print("Generating model...")
    model = Sequential()
    if PRETRAIN_EMBEDDINGS:
        model.add(Embedding(len(tokenizer.word_index) + 1,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_QUERY_LENGTH,
                            trainable=True))
    else:
        model.add(Embedding(len(tokenizer.word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_QUERY_LENGTH,
                            trainable=True))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=0.0001)
    early_stopper = EarlyStopping(patience=5)
    print("Compiling model...")
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print("Training model...")
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(x_val, y_val), verbose=2,
                        callbacks=[early_stopper],
                        class_weight=class_weights)

    print("Determining validation metrics...")
    val_prec, val_recall, val_f1 = evaluate_model(model, x_train, y_train, x_val, y_val)
    model_name = input("Enter a filename for model >> ")
    save_model(model, history, model_name,
               prec=val_prec, recall=val_recall, f1=val_f1,
               epochs=EPOCHS, batch_size=BATCH_SIZE, random_seed=SEED,
               class_weights=class_weights, sample_type=SAMPLE_TYPE)
    return model


def main():
    nn_bag_of_words()
    return



if __name__ == "__main__":
    main()