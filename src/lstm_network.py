from utils import *
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Embedding
from keras.models import Model
from keras.initializers import Constant

VAL_SPLIT = 0.2

def main():
    x, y = read_data_embeddings()
    x, y = shuffle_arrays(x, y, seed=20)
    num_val_examples = int(VAL_SPLIT * x.shape[0])
    x_train = x[:-num_val_examples]
    y_train = y[:-num_val_examples]

    x_val = x[-num_val_examples:]
    y_val = y[-num_val_examples:]
    return


if __name__ == "__main__":
    main()