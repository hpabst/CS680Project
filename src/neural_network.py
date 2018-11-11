from utils import *
from data_rebalancing import balance_samples
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, Dropout, Embedding, Flatten
from keras.models import Model, Sequential
from keras.initializers import Constant

VAL_SPLIT = 0.2
MAX_QUERY_LENGTH = 10
VOCAB_SIZE = 5000
EMBEDDING_DIM = 300
EMBEDDING_DIR = "../embeddings/glove.6B.300d.txt"

SEED = None
EPOCHS = 50
BATCH_SIZE = 2
WEIGHT_CLASSES = False
SAMPLE_TYPE = None
PRETRAIN_EMBEDDINGS = False


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
    model.add(Dense(2048, input_shape=(VOCAB_SIZE,), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print("Compiling model...")
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("Training model...")
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(X_val, y_val), verbose=2,
                        class_weight=class_weights)

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
    model.add(Dense(2048, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print("Compiling model...")
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("Training model...")
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(x_val, y_val), verbose=2,
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