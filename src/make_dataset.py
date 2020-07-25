import numpy as np
import pandas as pd
import os
import json

def main():
    df = pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk('./data/raw'):
        for filename in filenames:
            if filename.endswith('.json'): 
                file = open(os.path.join(dirpath, filename), 'r')
                df = pd.concat([df, pd.read_json(file)])
    print(f"Read {len(df)} number of samples")

if __name__ == '__main__':
    main()