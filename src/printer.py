import pandas as pd
import numpy as np

def print_general_statistics(df):
    print("Total training samples:", len(df), "\n")
    print("Partial data\n", df.iloc[0:4, 0:6], "\n")
    print("Maps in the data\n", pd.unique(df.map), "\n")
    print("Samples per map\n", df.groupby('map')['map'].count())
    df.map.describe()

def print_column_types(df):
    df.dtypes.value_counts()
    categorical_columns = df.select_dtypes('object').columns
    print(len(df.columns)-len(df.select_dtypes('object').columns),'numerical columns:')
    print([i for i in list(df.columns) if i not in list(df.select_dtypes('object').columns)], '\n')
    print(len(df.select_dtypes('object').columns),'categorical columns:')
    print(list(df.select_dtypes('object').columns))

def print_round_snapshot(round_index):
    round_snapshot = X.iloc[round_index]
    print(round_snapshot)
