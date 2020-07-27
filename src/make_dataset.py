import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def main():
    test_mode = False
    pd.reset_option('^display.', silent=True)
    file_path = ""

    # Load data
    if test_mode:
        file_path = './data/raw/dataset_initial/dataset_08.json'
        df = pd.read_json(file_path)
    else:
        df = load_data()

    print(f"Read {len(df)} number of samples")

    # # Make test/validation split
    # X = df.copy()
    # X_train, X_valid = make_test_split(X)
    # X_train.head()

    # get some statistics of the total data
    print_general_statistics(df)

    # print a round snapshot as a sample
    print_round_snapshot(25)

    # Show column types
    print_column_types(df)

    # Add ct_score and t_score columns from current_score
    X_train_cs = assign_scores(X_train).drop(['current_score'], axis=1)
    X_valid_cs = assign_scores(X_valid).drop(['current_score'], axis=1)

    # Use OH encoder to encode object cols, then remove and add enc
    object_cols = ['map', 'round_status']
    X_train_enc, X_valid_enc = encode_inputs(X_train_cs, X_valid_cs, object_cols)
    num_X_train = X_train_cs.drop(object_cols, axis=1)
    num_X_valid = X_valid_cs.drop(object_cols, axis=1)
    X_train = pd.concat([num_X_train, X_train_enc], axis=1)
    X_valid = pd.concat([num_X_valid, X_valid_enc], axis=1)

    # Encode targets
    y_train, y_valid = encode_targets(y_train, y_valid)

    # Flatten JSON structure in alive_players
    stats_cols = ['health', 'armor', 'has_helmet', 'has_defuser', 'money', 'team', 'inventory']
    for col in stats_cols:
        X_train = pd.merge(X_train, encode_players_stats(X_train, col), left_on='snapshot_id', right_on='snapshot_id', how='left')
        X_valid = pd.merge(X_valid, encode_players_stats(X_valid, col), left_on='snapshot_id', right_on='snapshot_id', how='left')
        X_train = X_train.drop([f'player_unknown_{col}'], axis=1, errors='ignore')
        X_valid = X_valid.drop([f'player_unknown_{col}'], axis=1, errors='ignore')
        
    # Drop alive_players as it has been encoded
    X_train = X_train.drop(['alive_players'], axis=1)
    X_valid = X_valid.drop(['alive_players'], axis=1)

    # Define player range
    pc_range = range(1, 11)

    # Handle NaN's for groups: health, armor, has_helmet, has_defuser, money
    col_reg = ['health', 'armor', 'has_helmet', 'has_defuser', 'money']
    for col in col_reg:
        for pc in pc_range:
                X_train[f'player_{pc}_{col}'].fillna(0, inplace=True)
                X_valid[f'player_{pc}_{col}'].fillna(0, inplace=True)     
    
    # Handle dead players for groups: team
    for pc in pc_range:
        X_train[f'player_{pc}_team'].fillna('Dead', inplace=True)
        X_valid[f'player_{pc}_team'].fillna('Dead', inplace=True)

    # Encode as 0/1's for groups: has_helmet, has_defuser
    col_labels = ['has_helmet', 'has_defuser']
    for col in col_labels:
        for pc in pc_range:
            X_train[f'player_{pc}_{col}'] = X_train[f'player_{pc}_{col}'].replace({True: 1, False: 0})
            X_valid[f'player_{pc}_{col}'] = X_valid[f'player_{pc}_{col}'].replace({True: 1, False: 0})
            
    # OH encode player teams
    player_cols = ['player_' + str(pc) + '_team' for pc in pc_range]
    X_train_enc, X_valid_enc = encode_inputs(X_train, X_valid, player_cols)
    num_X_train = X_train.drop(player_cols, axis=1)
    num_X_valid = X_valid.drop(player_cols, axis=1)
    X_train = pd.concat([num_X_train, X_train_enc], axis=1)
    X_valid = pd.concat([num_X_valid, X_valid_enc], axis=1)

    pd.core.frame.DataFrame.drop_suffix = drop_suffix

    for pc in pc_range:
        col = 'player_' + str(pc) + '_inventory'
        X_train = pd.merge(X_train, handle_nan_inventory(X_train[col]), left_index=True, right_index=True)
        X_valid = pd.merge(X_valid, handle_nan_inventory(X_valid[col]), left_index=True, right_index=True)
        X_train = X_train.drop([col + '_y'], axis=1)
        X_valid = X_valid.drop([col + '_y'], axis=1)
        X_train = X_train.drop_suffix('_x')
        X_valid = X_valid.drop_suffix('_x')

    # Handle a case where the cells are whitespace
    for pc in pc_range:
        col = 'player_' + str(pc) + '_inventory'
        text_empty_train = X_train[col].str.len() < 1
        text_empty_valid = X_valid[col].str.len() < 1
        
        indices_train = X_train.loc[text_empty_train].index
        indices_valid = X_valid.loc[text_empty_valid].index
        
        for idx in indices_train:
            X_train.at[idx, col] = [{'item_type': 'None', 'clip_ammo': 0, 'reserve_ammo': 0}]
        for idx in indices_valid:
            X_valid.at[idx, col] = [{'item_type':'None', 'clip_ammo':0, 'reserve_ammo':0}]

    # Encode player's inventory
    def encode_players_inventory(df, pc):
        col = 'player_' + str(pc) + '_inventory'
        inv_list = [pd.DataFrame(inv) for inv in df[col]]
        pc_df = pd.concat(inv_list, axis=1, keys=df.index).transpose().reset_index(level=[1]).loc[lambda df: df['level_1'] == 'item_type'].rename(
            columns={0: 'player_' + str(pc) + '_weapon_1',
                    1: 'player_' + str(pc) + '_weapon_2',
                    2: 'player_' + str(pc) + '_grenade_1',
                    3: 'player_' + str(pc) + '_grenade_2',
                    4: 'player_' + str(pc) + '_grenade_3',
                    5: 'player_' + str(pc) + '_grenade_4',
                    6: 'player_' + str(pc) + '_grenade_5'}).drop(['level_1'], axis=1)
        return pc_df

    for pc in pc_range:
        X_train = pd.merge(X_train, encode_players_inventory(X_train, pc), left_index=True, right_index=True)
        X_valid = pd.merge(X_valid, encode_players_inventory(X_valid, pc), left_index=True, right_index=True)
        X_train = X_train.drop([f'player_{str(pc)}_inventory'], axis=1)
        X_valid = X_valid.drop([f'player_{str(pc)}_inventory'], axis=1)
    
    # Handle NaN's for weapons and grenades
    X_train = X_train.fillna('None')
    X_valid = X_valid.fillna('None')

    # Do OH encoding of weapons and grenades
    import itertools
    col_inv = ['weapon_1', 'weapon_2', 'grenade_1', 'grenade_2', 'grenade_3', 'grenade_4']
    player_cols = [['player_' + str(pc) + '_' + str(ci) for ci in col_inv] for pc in pc_range]
    player_cols = pd.Series(itertools.chain(*player_cols))
    X_train_enc, X_valid_enc = encode_inputs(X_train, X_valid, player_cols)
    num_X_train = X_train.drop(player_cols, axis=1)
    num_X_valid = X_valid.drop(player_cols, axis=1)
    X_train = pd.concat([num_X_train, X_train_enc], axis=1)
    X_valid = pd.concat([num_X_valid, X_valid_enc], axis=1)

    # Remove duplicate columns
    X_train = X_train.loc[:,~X_train.columns.duplicated()]
    X_valid = X_valid.loc[:, ~X_valid.columns.duplicated()]
    
    # Save encoded results
    X_train.to_csv("../interim/x_train.csv", index=False)
    X_valid.to_csv("../interim/x_valid.csv", index=False)
    pd.Series(y_train).to_csv("../interim/y_train.csv", index=False)
    pd.Series(y_valid).to_csv("../interim/y_valid.csv", index=False)

def load_data(path):
    df = pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk('./data/raw'):
        for filename in filenames:
            if filename.endswith('.json'): 
                file = open(os.path.join(dirpath, filename), 'r')
                df = pd.concat([df, pd.read_json(file)])
    return df

def make_test_split(df):
    # Sample data
    X = X.sample(frac=1,random_state=0).reset_index(drop=True)

    # Set a unique snapshot id
    X = X.assign(snapshot_id=(X.index).astype(int))
                
    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['round_winner'], inplace=True)
    y = X.round_winner
    X.drop(['round_winner'], axis=1, inplace=True)

    # Drop cols not targeted for model
    cols_to_drop = ['active_smokes', 'active_molotovs', 'previous_kills']
    X.drop(cols_to_drop, axis=1, inplace=True)

    # Drop cols with missing values
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X.drop(cols_with_missing, axis=1, inplace=True)

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Reset the index
    X_train = X_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    return (X_train, X_valid)

def print_general_statistics(df):
    print("Total training samples:", len(df), "\n")
    print("Partial data\n", df.iloc[0:4, 0:6], "\n")
    print("Maps in the data\n", pd.unique(df.map), "\n")
    print("Samples per map\n", df.groupby('map')['map'].count())
    df.map.describe()

# print a round snapshot as a sample
def print_round_snapshot(round_index = 25):
    round_snapshot = X.iloc[round_index]
    print(round_snapshot)

def print_column_types(df):
    df.dtypes.value_counts()
    categorical_columns = df.select_dtypes('object').columns
    print(len(df.columns)-len(df.select_dtypes('object').columns),'numerical columns:')
    print([i for i in list(df.columns) if i not in list(df.select_dtypes('object').columns)], '\n')
    print(len(df.select_dtypes('object').columns),'categorical columns:')
    print(list(df.select_dtypes('object').columns))

def assign_scores(df):
    df = df.assign(ct_score=(df['current_score'].str[0]).astype(int))
    df = df.assign(t_score=(df['current_score'].str[1]).astype(int))
    return df

def encode_inputs(X_train, X_valid, object_cols):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_train_enc = pd.DataFrame(ohe.fit_transform(X_train[object_cols]))
    X_valid_enc = pd.DataFrame(ohe.transform(X_valid[object_cols]))
    X_train_enc.columns = ohe.get_feature_names(object_cols)
    X_valid_enc.columns = ohe.get_feature_names(object_cols)
    X_train_enc.index = X_train.index
    X_valid_enc.index = X_valid.index
    return X_train_enc, X_valid_enc

def encode_targets(y_train, y_valid):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_valid_enc = le.transform(y_valid)
    return y_train_enc, y_valid_enc

# Encode player's stats
def encode_players_stats(df, col):
    df = df.set_index(['snapshot_id'])['alive_players']
    alive_players_list = [pd.DataFrame(alive_player) for alive_player in df]
    
    df = (pd.concat(alive_players_list, axis=1, keys=df.index)
          .transpose()
          .reset_index(level=[0,1])
          .rename(columns={'level_1':'player_stats'}))
    df = df.rename(columns={0:'player_1', 1:'player_2', 2:'player_3', 3:'player_4', 4:'player_5',
                  5:'player_6', 6:'player_7', 7:'player_8', 8:'player_9', 9:'player_10', 10:'player_unknown'})
    
    df = df[df.player_stats == col].reset_index(drop=True)
    df = df.drop(['player_stats'], axis=1)
    df.columns = ['{}{}'.format(c, '' if c in ['snapshot_id'] else '_' + str(col)) for c in df.columns]
    return df

# Handle NaN's for inventory
def handle_nan_inventory(df):
    nan_inv = pd.isna(df)
    nan_entries = list(nan_inv[nan_inv == True].index)
    for idx in nan_entries:
        df.iloc[idx] = [{'item_type': 'None', 'clip_ammo': 0, 'reserve_ammo': 0}] 
    return df

def drop_suffix(self, suffix):
    self.columns = self.columns.str.rstrip(suffix)
    return self

if __name__ == '__main__':
    main()