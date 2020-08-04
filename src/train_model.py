import numpy as np
import pandas as pd
import file_reader as fr
import file_writer as fw
import config as cfg
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

def main():
    X = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'X_full.csv')
    y = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'y_full.csv')

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
     test_size=0.2, random_state=0, stratify=y)

    model = make_model()
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5)
    tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())
    
    history = model.fit(np.array(X_train), np.array(y_train), epochs=100,
     validation_data=(np.array(X_valid), np.array(y_valid)),
      callbacks=[early_stopping_cb, tensorboard_cb])
    
def make_model():
    model = keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(93, activation="selu", kernel_initializer="lecun_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model
 
def scale_data(X_train, X_valid):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    return X_train, X_valid

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(cfg.LOGS_DIR, run_id)

if __name__ == '__main__':
    main()