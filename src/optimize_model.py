import numpy as np
import pandas as pd
import file_reader as fr
import file_writer as fw
import config as cfg
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, KernelPCA

def main():
    X = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'X_full.csv')
    y = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'y_full.csv')

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y,
     test_size=0.1, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full,
     test_size=0.25, random_state=0)

    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of validation samples: {X_valid.shape[0]}")
    print(f"Number of testing samples: {X_test.shape[0]}")

    keras_pipeline = Pipeline([
        ("scaler", MinMaxScaler(feature_range=(-1, 1))),
        ("clf", keras.wrappers.scikit_learn.KerasClassifier(build_fn=make_model,
         n_input=X_train.shape[1], n_class=y_train.shape[1]))
    ])
    
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5)
    tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())

    network_layers_opts = [[64], [64, 32], [64, 32, 16],
     [128], [128, 64], [128, 64, 32], [256, 128, 64, 32]]
    param_grid = {'clf__network_layers': network_layers_opts,
     'clf__epochs': [10],
     'clf__dropout_rate': [0.5, 0.4, 0.3, 0.2, 0.1, 0],
     'clf__optimizer': ['Nadam'],
     'clf__activation': ['selu'],
     'clf__k_initializer': ['lecun_uniform'],
     'clf__l1_penalty': [0.01, 0.1, 0.2, 0.5]
    }

    rs_keras = RandomizedSearchCV(keras_pipeline, param_distributions=param_grid,
     cv=5, refit=False, verbose=10, scoring="accuracy")
    result = rs_keras.fit(np.array(X_train), np.array(y_train))
    
    print("Best: %f using %s" % (result.best_score_, result.best_params_))
    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
def make_model(network_layers=[64], dropout_rate=0, optimizer="Nadam",
 activation="selu", k_initializer='lecun_uniform', l1_penalty=0,
  n_input=93, n_class=1):
    model = keras.models.Sequential()

    for index, layers in enumerate(network_layers):
        if not index:
            model.add(keras.layers.Dense(layers, input_dim=n_input, activation=activation,
             kernel_initializer=k_initializer, kernel_regularizer = keras.regularizers.l1(l1_penalty)))
        else:
            model.add(keras.layers.Dense(layers, kernel_initializer=k_initializer,
             activation=activation, kernel_regularizer = keras.regularizers.l1(l1_penalty)))
        if dropout_rate and index:
            model.add(keras.layers.AlphaDropout(dropout_rate))

    model.add(keras.layers.Dense(n_class, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
 
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(cfg.LOGS_DIR, run_id)

if __name__ == '__main__':
    main()