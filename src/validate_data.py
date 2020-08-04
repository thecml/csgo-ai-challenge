import numpy as np
import pandas as pd
import file_reader as fr
import file_writer as fw
import config as cfg

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    X = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'X_full.csv')
    y = fr.read_csv(cfg.PROCESSED_DATA_DIR, 'y_full.csv')

    print(X.isnull().values.any())
    print(y.isnull().values.any())

    print(X.describe())
    print(y.describe())

    print(X.dtypes)
    print(y.dtypes)