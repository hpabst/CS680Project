from utils import *
from metrics import precision, recall, fbeta_score
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Embedding, Input
from keras.models import Model, Sequential
from keras.initializers import Constant

VAL_SPLIT = 0.2
EMBEDDING_DIM = 300
MAX_QUERY_LENGTH = 10
EMBEDDING_DIR = "../embeddings/glove.6B.300d.txt"

def main():
    print("Reading data...")
    x, y, tokenizer = read_data_embeddings(max_input_length=MAX_QUERY_LENGTH)
    x, y = shuffle_arrays(x, y, seed=20)
    num_val_examples = int(VAL_SPLIT * x.shape[0])
    x_train = x[:-num_val_examples]
    y_train = y[:-num_val_examples]

    x_val = x[-num_val_examples:]
    y_val = y[-num_val_examples:]
    print("Creating word embeddings matrix...")
    embedding_matrix = create_embedding_matrix(EMBEDDING_DIR,
                                               tokenizer.word_index,
                                               EMBEDDING_DIM)
    print("Generating model...")
    model = Sequential()
    #model.add(Input(shape=(MAX_QUERY_LENGTH,), dtype='int32'))
    model.add(Embedding(len(tokenizer.word_index) + 1,
                        EMBEDDING_DIM,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=MAX_QUERY_LENGTH,
                        trainable=False))
    model.add(LSTM(128, dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    print("Compiling model...")
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print("Training model...")
    history = model.fit(x_train, y_train, batch_size=128, epochs=100,
                        validation_data=(x_val, y_val))
    print("Determining validation metrics...")
    val_metrics = model.evaluate(x=x_val, y=y_val)
    print(val_metrics)
    evaluate_model(model, x_train, y_train, x_val, y_val)

    return


if __name__ == "__main__":
    main()