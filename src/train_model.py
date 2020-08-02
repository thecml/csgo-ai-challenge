import numpy as np
import pandas as pd
import src.file_reader as fr
import src.file_writer as fw
import src.config as cfg
import tensorflow as tf
from tensorflow import keras

def main():
    tf.test.is_built_with_cuda()
    nrows = 100
    X_train = fr.read_csv(cfg.INTERIM_DATA_DIR, 'X_train.csv', nrows=nrows)
    X_valid = fr.read_csv(cfg.INTERIM_DATA_DIR, 'X_valid.csv', nrows=nrows)
    y_train = fr.read_csv(cfg.INTERIM_DATA_DIR, 'y_train.csv', nrows=nrows)
    y_valid = fr.read_csv(cfg.INTERIM_DATA_DIR, 'y_valid.csv', nrows=nrows)

    model = make_model()

def make_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    main()