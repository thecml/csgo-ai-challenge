import numpy as np
import pandas as pd
import src.lib.file_reader as file_reader
import src.lib.file_writer as file_writer
import src.config as cfg

def main():
    nrows = None
    X_train = file_reader.read_csv(cfg.INTERIM_DATA_DIR, 'X_train.csv', nrows=nrows)
    X_valid = file_reader.read_csv(cfg.INTERIM_DATA_DIR, 'X_valid.csv', nrows=nrows)
    y_train = file_reader.read_csv(cfg.INTERIM_DATA_DIR, 'y_train.csv', nrows=nrows)
    y_valid = file_reader.read_csv(cfg.INTERIM_DATA_DIR, 'y_valid.csv', nrows=nrows)

    X_train = prepare_data(X_train)
    X_valid = prepare_data(X_valid)

    file_writer.write_csv(X_train, cfg.PROCESSED_DATA_DIR, 'X_train.csv')
    file_writer.write_csv(X_valid, cfg.PROCESSED_DATA_DIR, 'X_valid.csv')

def prepare_data(df):
    cols_leftover = '7_x|8_x|7_y|8_y'
    cols_grenade = 'Grenade|grenade|Flashbang|C4'
    cols_pistol = 'Deagle|FiveSeven|Clock|UspS|P250|P2000|Tec9'
    cols_misc = 'None|Dead|snapshot_id'

    df = df.drop(df.columns[df.columns.str.contains(cols_leftover)], axis=1)
    df = df.drop(df.columns[df.columns.str.contains(cols_grenade)], axis=1)
    df = df.drop(df.columns[df.columns.str.contains(cols_pistol)], axis=1)
    df = df.drop(df.columns[df.columns.str.contains(cols_misc)], axis=1)

    df['round_status_time_left'] = df['round_status_time_left'].apply(
        lambda x: np.around(x, decimals=3))
    

    
    return df

if __name__ == '__main__':
    main()