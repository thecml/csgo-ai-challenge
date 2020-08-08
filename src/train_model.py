import numpy as np
import pandas as pd
import file_reader as fr
import file_writer as fw
import config as cfg
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import power_transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

def main():
    X = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'X_full.csv')
    y = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'y_full.csv')

    #X = prepare_data(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y,
     test_size=0.1, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full,
     test_size=0.25, random_state=0)

    model = make_model()
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10)
    tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())

    history = model.fit(np.array(X_train), np.array(y_train), epochs=100,
     validation_data=(np.array(X_valid), np.array(y_valid)),
      callbacks=[early_stopping_cb, tensorboard_cb])
    results = model.evaluate(X_test, y_test)
    print("Test loss, test acc:", results)
    
def make_model():
    model = keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("elu"))
    model.add(keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("elu"))
    model.add(keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("elu"))
    model.add(keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("elu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model
 
def prepare_data(df):
    # Add 1m55s to round if freezetime
    df['round_status_time_left'] = df.apply(add_round_time, axis=1)
    df = df.drop(['round_status_FreezeTime', 'round_status_Normal'], axis=1)

    # Clip roundtime value
    df['round_status_time_left'] = df['round_status_time_left'].clip(0, 175)

    # Drop map columns
    df = df.drop(df.columns[df.columns.str.contains('map')], axis=1)

    # Drop pistol columns
    pistol_cols = 'Cz75Auto|Elite|Glock|P250|UspS|P2000|Tec9|FiveSeven'
    df = df.drop(df.columns[df.columns.str.contains(pistol_cols)], axis=1)

    # Make data more Gaussian-like
    cols = ['round_status_time_left', 'ct_money', 't_money', 'ct_health',
     't_health', 'ct_armor', 't_armor', 'ct_helmets', 't_helmets',
      'ct_defuse_kits', 'ct_players', 't_players']
    for col in cols:
        df[col] = yeo_johnson(df[col])

    return df

def scale_data(X_train, X_valid, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    return X_train, X_valid, X_test

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(cfg.LOGS_DIR, run_id)

def yeo_johnson(series):
    arr = np.array(series).reshape(-1, 1)
    return power_transform(arr, method='yeo-johnson')

def add_round_time(row):
    if row['round_status_FreezeTime'] == 1:
        return row['round_status_time_left'] + 155
    else:
        return row['round_status_time_left']

if __name__ == '__main__':
    main()