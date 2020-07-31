from pathlib import Path
import pandas as pd
import os

def read_csv(path, filename):
    return pd.read_csv(Path.joinpath(path, filename))

def read_json_data():
    df = pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk('./data/raw'):
        for filename in filenames:
            if filename.endswith('.json'): 
                file = open(os.path.join(dirpath, filename), 'r')
                df = pd.concat([df, pd.read_json(file)])
    return df