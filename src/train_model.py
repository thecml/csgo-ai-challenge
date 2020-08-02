import numpy as np
import pandas as pd
import src.file_reader as fr
import src.file_writer as fw
import src.config as cfg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

def main():
    X_train = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'X_train.csv')
    X_valid = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'X_valid.csv')
    y_train = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'y_train.csv')
    y_valid = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'y_valid.csv')

    X_train, X_valid = scale_data(X_train, X_valid)

    model = make_model()

    history = model.fit(X_train, np.array(y_train), epochs=20,
     validation_data=(X_valid, np.array(y_valid)))

    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

def make_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def scale_data(X_train, X_valid):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    return (X_train, X_valid)

if __name__ == '__main__':
    main()