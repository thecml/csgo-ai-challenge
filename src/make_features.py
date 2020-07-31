import numpy as np
import pandas as pd
import os
import src.file_reader as fr
import src.file_writer as fw
import src.config as cfg

def main():
    X_train = fr.read_csv(cfg.INTERIM_DATA_DIR, 'X_train.csv')
    X_valid = fr.read_csv(cfg.INTERIM_DATA_DIR, 'X_valid.csv')

if __name__ == '__main__':
    main()