import numpy as np
import pandas as pd
import src.lib.file_reader as file_reader
import src.lib.file_writer as file_writer
import src.config as cfg

def main():
    X_train = file_reader.read_csv(cfg.INTERIM_DATA_DIR, 'X_train.csv', nrows=1000)
    X_valid = file_reader.read_csv(cfg.INTERIM_DATA_DIR, 'X_valid.csv', nrows=1000)
    y_train = file_reader.read_csv(cfg.INTERIM_DATA_DIR, 'y_train.csv', nrows=1000)
    y_valid = file_reader.read_csv(cfg.INTERIM_DATA_DIR, 'y_valid.csv', nrows=1000)

    X_train = prepare_data(X_train)
    X_valid = prepare_data(X_valid)

    file_writer.write_csv(X_train, cfg.INTERIM_DATA_DIR, 'X_train_1000.csv')
    file_writer.write_csv(y_train, cfg.INTERIM_DATA_DIR, 'y_train_1000.csv')


def prepare_data(df):
    cols_leftover = '7_x|8_x|7_y|8_y|'
    cols_grenade = 'Dead|Grenade|grenade|Flashbang|C4'
    cols_pistol = 'None|Deagle|FiveSeven|Clock|UspS|P250|P2000|Tec9'
    cols_snapshot = 'snapshot_id'

    df = df.drop(df.columns[df.columns.str.contains(cols_leftover)], axis=1)
    df = df.drop(df.columns[df.columns.str.contains(cols_grenade)], axis=1)
    df = df.drop(df.columns[df.columns.str.contains(cols_pistol)], axis=1)
    df = df.drop(df.columns[df.columns.str.contains(cols_snapshot)], axis=1)
    
    return df

if __name__ == '__main__':
    main()