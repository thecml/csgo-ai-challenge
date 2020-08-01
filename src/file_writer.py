from pathlib import Path
import pandas as pd

def write_csv(df, path, file_name, index=False):
    df.to_csv(Path.joinpath(path, file_name), index=index)