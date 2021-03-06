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
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool

def main():
    X = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'X_full.csv')
    y = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'y_full.csv')

    X = prepare_data(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y,
     stratify=y, test_size=0.1, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full,
     stratify=y_train_full, test_size=0.25, random_state=0)

    keras_shallow = make_keras_model(n_layers=4)
    keras_deep = make_keras_model(n_layers=8, regularized=False, dropout=True)
    keras_dropout = make_keras_model(n_layers=4, regularized=False, dropout=True)

    tensorboard_cb_shallow = keras.callbacks.TensorBoard(get_run_logdir('shallow'))
    tensorboard_cb_deep = keras.callbacks.TensorBoard(get_run_logdir('deep'))
    tensorboard_cb_dropout = keras.callbacks.TensorBoard(get_run_logdir('dropout'))
    reduce_lr_cb_shallow = get_keras_reducelr_cb()
    reduce_lr_cb_deep = get_keras_reducelr_cb()
    reduce_lr_cb_dropout = get_keras_reducelr_cb()

    model_checkpoint_cb_shallow = keras.callbacks.ModelCheckpoint(get_run_modeldir('shallow'))
    model_checkpoint_cb_deep = keras.callbacks.ModelCheckpoint(get_run_modeldir('deep'))
    model_checkpoint_cb_dropout = keras.callbacks.ModelCheckpoint(get_run_modeldir('dropout'))

    history_shallow = keras_shallow.fit(np.array(X_train), np.array(y_train), epochs=1, #Set to 200
     validation_data=(np.array(X_valid), np.array(y_valid)),
      callbacks=[tensorboard_cb_shallow, model_checkpoint_cb_shallow, reduce_lr_cb_shallow])
    
    history_deep = keras_deep.fit(np.array(X_train), np.array(y_train), epochs=1, #Set to 250
     validation_data=(np.array(X_valid), np.array(y_valid)),
      callbacks=[tensorboard_cb_deep, model_checkpoint_cb_deep, reduce_lr_cb_deep])

    history_dropout = keras_dropout.fit(np.array(X_train), np.array(y_train), epochs=1, #Set to 250
     validation_data=(np.array(X_valid), np.array(y_valid)),
      callbacks=[tensorboard_cb_dropout, model_checkpoint_cb_dropout, reduce_lr_cb_dropout])

    results_shallow = keras_shallow.evaluate(X_test, y_test)
    results_deep = keras_deep.evaluate(X_test, y_test)
    results_dropout = keras_dropout.evaluate(X_test, y_test)

    print("shallow acc:", results_shallow)
    print("deep acc:", results_deep)
    print("dropout acc:", results_dropout)

    clf_cat = CatBoostClassifier(learning_rate=0.03,
                           iterations=20000,
                           task_type="GPU",
                           devices='0:1',
                           eval_metric='AUC')
    clf_cat.fit(Pool(X_train, y_train.values.ravel()),
     eval_set=Pool(X_valid, y_valid.values.ravel()),
      early_stopping_rounds=200)
    results_cat = clf_cat.predict(X_test)
    print(f"cat acc: {accuracy_score(y_test, results_cat)}")

    clf_xgb = make_xgb()
    clf_xgb.fit(X_train, y_train.values.ravel(),
     eval_set=[(X_train, y_train.values.ravel()), (X_valid, y_valid.values.ravel())], 
            eval_metric='auc', verbose=200, early_stopping_rounds=200)
    results_xgb = clf_xgb.predict(X_test)
    print(f"xgb acc: {accuracy_score(y_test, results_xgb)}")

    clf_knn = make_knn()
    clf_knn.fit(X_train, y_train.values.ravel())
    results_knn = clf_knn.predict(X_test)
    print(f"knn acc: {accuracy_score(y_test, results_knn)}")

def make_keras_model(n_layers, regularized=False, dropout=False):
    model = keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    for n in range(n_layers):
        if regularized:
            model.add(keras.layers.Dense(300, kernel_initializer="he_normal",
             kernel_regularizer=keras.regularizers.l1(0.01), use_bias=False))
        else:
            model.add(keras.layers.Dense(300,
             kernel_initializer="he_normal", use_bias=False))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("elu"))
        if dropout:
            model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model

def make_knn():
    return KNeighborsClassifier()

def make_lgbm():
    return LGBMClassifier(colsample_bytree=0.8, min_child_samples=214, 
    min_child_weight=0.001, num_leaves=39, reg_alpha=1, reg_lambda=0.1,
    subsample=0.51, random_state=0)

def make_xgb():
    return XGBClassifier(colsample_bytree=0.62, learning_rate=0.1,
     max_depth=10, min_child_weight=0.1, n_estimators=5000,
     reg_alpha=1, reg_lambda=5, subsample=0.96, random_state=0)

def get_keras_reducelr_cb():
    return keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
     patience=5, min_lr=0.001)

def prepare_data(df):
    # Add 1m55s to round if freezetime
    df['round_status_time_left'] = df.apply(add_round_time, axis=1)
    df = df.drop(['round_status_FreezeTime', 'round_status_Normal'], axis=1)

    # Clip roundtime value
    df['round_status_time_left'] = df['round_status_time_left'].clip(0, 175)

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

def get_run_logdir(name):
    run_id = time.strftime(f"run_%Y_%m_%d-%H_%M_%S_{name}")
    return os.path.join(cfg.LOGS_DIR, run_id)

def get_run_modeldir(name):
    run_id = time.strftime(f"run_%Y_%m_%d-%H_%M_%S_{name}")
    return os.path.join(cfg.MODELS_DIR, run_id)

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