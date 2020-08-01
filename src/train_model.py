import numpy as np
import pandas as pd
import src.lib.file_reader as fr
import src.lib.file_writer as fw
import src.config as cfg

def main():
    nrows = 100
    X_train = fr.read_csv(cfg.INTERIM_DATA_DIR, 'X_train.csv', nrows=nrows)
    X_valid = fr.read_csv(cfg.INTERIM_DATA_DIR, 'X_valid.csv', nrows=nrows)
    y_train = fr.read_csv(cfg.INTERIM_DATA_DIR, 'y_train.csv', nrows=nrows)
    y_valid = fr.read_csv(cfg.INTERIM_DATA_DIR, 'y_valid.csv', nrows=nrows)

    X_train = prepare_data(X_train)
    X_valid = prepare_data(X_valid)

    fw.write_csv(X_train, cfg.PROCESSED_DATA_DIR, 'X_train.csv')
    fw.write_csv(X_valid, cfg.PROCESSED_DATA_DIR, 'X_valid.csv')
    fw.write_csv(y_train, cfg.PROCESSED_DATA_DIR, 'y_train.csv')
    fw.write_csv(y_valid, cfg.PROCESSED_DATA_DIR, 'y_valid.csv')

def prepare_data(df):
    cols_leftover = '7_x|8_x|7_y|8_y'
    cols_grenade = 'Grenade|grenade|Flashbang|C4'
    cols_pistol = 'Deagle|FiveSeven|Clock|UspS|P250|P2000|Tec9'
    cols_misc = 'None|Dead|Zeus|snapshot_id'

    # Delete a lot of unrelated information to wins
    df = df.drop(df.columns[df.columns.str.contains(cols_leftover)], axis=1)
    df = df.drop(df.columns[df.columns.str.contains(cols_grenade)], axis=1)
    df = df.drop(df.columns[df.columns.str.contains(cols_pistol)], axis=1)
    df = df.drop(df.columns[df.columns.str.contains(cols_misc)], axis=1)

    # Cut off some decimals from round time
    df['round_status_time_left'] = df['round_status_time_left'].apply(
        lambda x: np.around(x, decimals=2))
    
    # Delete another leftover column
    if '7' in df.columns: del df['7']

    return df

if __name__ == '__main__':
    main()