from utils import *
from data_rebalancing import balance_samples
from metrics import *
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Embedding, Input
from keras.models import Model, Sequential
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

VAL_SPLIT = 0.2
EMBEDDING_DIM = 300
MAX_QUERY_LENGTH = 10
EMBEDDING_DIR = "../embeddings/glove.6B.300d.txt"

SEED = None
EPOCHS = 50
BATCH_SIZE = 64
WEIGHT_CLASSES = False
SAMPLE_TYPE = None
PRETRAIN_EMBEDDINGS = False


def main():
    print("Reading data...")
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
    model.add(LSTM(128, dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=0.0001)
    early_stopper = EarlyStopping(patience=5)
    print("Compiling model...")
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print("Training model...")
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(x_val, y_val),
                        verbose=2,
                        class_weight=class_weights,
                        callbacks=[early_stopper])
    print("Determining validation metrics...")
    val_prec, val_recall, val_f1 = evaluate_model(model, x_train, y_train, x_val, y_val)
    model_name = input("Enter a filename for model >>")
    save_model(model, history, model_name,
               prec=val_prec, recall=val_recall, f1=val_f1,
               epochs=EPOCHS, batch_size=BATCH_SIZE, random_seed=SEED,
               class_weights=class_weights, sample_type=SAMPLE_TYPE)
    return model


if __name__ == "__main__":
    main()