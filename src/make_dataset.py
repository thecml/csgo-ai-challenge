#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import src.file_reader as fr
import src.file_writer as fw
import src.config as cfg

def main():
    df = fr.read_json_data()

    # Keep 10000 rows
    df = df[:10000]

    # Make test/validation split
    X = df.copy()
    X_train, X_valid, y_train, y_valid = make_test_split(df, test_size=0.2)

    # Add ct_score and t_score columns from current_score
    X_train_cs = assign_scores(X_train).drop(['current_score'], axis=1)
    X_valid_cs = assign_scores(X_valid).drop(['current_score'], axis=1)

    # Use OH encoder to encode object cols
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

    # Handle NaN's for groups: health, armor, has_helmet, has_defuser, money
    pc_range = range(1, 11)
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

    # Create func to drop suffix
    pd.core.frame.DataFrame.drop_suffix = drop_suffix

    # Handle nan inventory
    for pc in pc_range:
        col = 'player_' + str(pc) + '_inventory'
        X_train = pd.merge(X_train, handle_nan_inventory(X_train[col]), left_index=True, right_index=True)
        X_valid = pd.merge(X_valid, handle_nan_inventory(X_valid[col]), left_index=True, right_index=True)
        X_train = X_train.drop([col + '_y'], axis=1)
        X_valid = X_valid.drop([col + '_y'], axis=1)
        X_train = X_train.drop_suffix('_x')
        X_valid = X_valid.drop_suffix('_x')

    # Handle a case where cells occur as whitespace
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
    col_inv = ['weapon_1', 'weapon_2', 'grenade_1', 'grenade_2', 'grenade_3', 'grenade_4', 'grenade_5']
    player_cols = [['player_' + str(pc) + '_' + str(ci) for ci in col_inv] for pc in pc_range]
    player_cols = pd.Series(itertools.chain(*player_cols))
    X_train_enc, X_valid_enc = encode_inputs(X_train, X_valid, player_cols)
    num_X_train = X_train.drop(player_cols, axis=1)
    num_X_valid = X_valid.drop(player_cols, axis=1)
    X_train = pd.concat([num_X_train, X_train_enc], axis=1)
    X_valid = pd.concat([num_X_valid, X_valid_enc], axis=1)

    # Remove duplicated columns
    X_train = X_train.loc[:,~X_train.columns.duplicated()]
    X_valid = X_valid.loc[:, ~X_valid.columns.duplicated()]

    # Prepare the data for modelling
    X_train = prepare_data(X_train)
    X_valid = prepare_data(X_valid)

    # Save encoded results
    fw.write_csv(X_train, cfg.PROCESSED_DATA_DIR, 'X_train.csv')
    fw.write_csv(X_valid, cfg.PROCESSED_DATA_DIR, 'X_valid.csv')
    fw.write_csv(pd.Series(y_train), cfg.PROCESSED_DATA_DIR, 'y_train.csv')
    fw.write_csv(pd.Series(y_valid), cfg.PROCESSED_DATA_DIR, 'y_valid.csv')

def make_test_split(X, test_size):
    # Sample data randomly
    #X = X.sample(frac=1, random_state=0).reset_index(drop=True)
    X = X.reset_index(drop=True)

    # Set a unique snapshot id
    X = X.assign(snapshot_id=(X.index).astype(int))
                
    # Split X and y
    y = X.round_winner
    X = X.drop(['round_winner'], axis=1)

    # Drop cols not targeted for model
    cols_to_drop = ['active_smokes', 'active_molotovs', 'previous_kills']
    X.drop(cols_to_drop, axis=1, inplace=True)

    # Drop cols with missing values
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X.drop(cols_with_missing, axis=1, inplace=True)

    # Break off validation set from training set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
     test_size=test_size, random_state=0)

    # Reset the index
    X_train = X_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    return (X_train, X_valid, y_train, y_valid)

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
    list(le.classes_)
    y_train_enc = le.transform(y_train)
    y_valid_enc = le.transform(y_valid)
    return y_train_enc, y_valid_enc

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

def handle_nan_inventory(df):
    nan_inv = pd.isna(df)
    nan_entries = list(nan_inv[nan_inv == True].index)
    for idx in nan_entries:
        df.iloc[idx] = [{'item_type': 'None', 'clip_ammo': 0, 'reserve_ammo': 0}] 
    return df

def drop_suffix(self, suffix):
    self.columns = self.columns.str.rstrip(suffix)
    return self

def prepare_data(df):
    cols_leftover = '7_x|8_x|7_y|8_y|'
    cols_grenade = 'Grenade|grenade|Flashbang|C4|'
    cols_misc = 'None|Dead|Zeus|snapshot_id'
    all_cols = cols_leftover + cols_grenade + cols_misc
    df = df.drop(df.columns[df.columns.str.contains(all_cols)], axis=1)

    # Delete another leftover column
    if '7' in df.columns: del df['7']

    # Cut off some decimals from round time
    df['round_status_time_left'] = df['round_status_time_left'].apply(
        lambda x: np.around(x, decimals=2))

    # Sum the stats of the teams
    numPlayers = 10
    df['ct_health'] = sum_player_cols(df, 'health', 'CT')
    df['t_health'] = sum_player_cols(df, 'health', 'Terrorist')
    df['ct_armor'] = sum_player_cols(df, 'armor', 'CT')
    df['t_armor'] = sum_player_cols(df, 'armor', 'Terrorist')
    df['ct_money'] = sum_player_cols(df, 'money', 'CT')
    df['t_money'] = sum_player_cols(df, 'money', 'Terrorist')
    df['ct_helmets'] = sum_player_cols(df, 'has_helmet', 'CT')
    df['t_helmets'] = sum_player_cols(df, 'has_helmet', 'Terrorist')
    df['ct_defuse_kits'] = sum_player_cols(df, 'has_defuser', 'CT')
    df['ct_players'] = sum([df[str(f'player_{i}_team_CT')] for i in range(1, numPlayers + 1)])
    df['t_players'] = sum([df[str(f'player_{i}_team_Terrorist')] for i in range(1, numPlayers + 1)])

    # Sum the weapons of the teams
    for weapon in cfg.weapon_list:
        df[f'ct_weapon_{weapon}'] = sum_player_cols(df, f'weapon_1_{weapon}', 'CT') \
         + sum_player_cols(df, f'weapon_2_{weapon}', 'CT')
        df[f't_weapon_{weapon}'] = sum_player_cols(df, f'weapon_1_{weapon}', 'Terrorist') \
         + sum_player_cols(df, f'weapon_2_{weapon}', 'Terrorist')
        
    # Drop individual player columns
    df = df.drop(df.columns[df.columns.str.contains('player_')], axis=1)

    return df

def sum_player_cols(df, name, team, numPlayers=10):
    return sum([df[str(f'player_{i}_' + name)].where(df[str(f'player_{i}_team_' + team)] == 1.0, 0)
     for i in range(1, numPlayers + 1) if str(f'player_{i}_' + name) in df.columns])

if __name__ == '__main__':
    main()