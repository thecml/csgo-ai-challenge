# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split

def main():
    df = pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk('./data/raw'):
        for filename in filenames:
            if filename.endswith('.json'): 
                file = open(os.path.join(dirpath, filename), 'r')
                df = pd.concat([df, pd.read_json(file)])
    print("Finished reading csv")
    print(len(df))

if __name__ == '__main__':
    main()