import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import file_reader as fr
import file_writer as fw
import config as cfg

def main():
    # Load the data
    df = fr.read_json_data()
    X = df.copy()
    X = X.reset_index(drop=True)
    X = X.assign(snapshot_id=(X.index).astype(int))
                
    # Split X and y
    y = X.round_winner
    X = X.drop(['round_winner'], axis=1)

    # Drop unused cols and those with NaN's
    cols_to_drop = ['active_smokes', 'active_molotovs', 'previous_kills']
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X.drop(cols_to_drop, axis=1, inplace=True)
    X.drop(cols_with_missing, axis=1, inplace=True)

    # Add ct_score and t_score columns from current_score
    X = assign_scores(X).drop(['current_score'], axis=1)

    # Use OH encoder to encode object cols
    object_cols = ['map', 'round_status']
    X_encoded = encode_inputs(X, object_cols)
    numerical_X = X.drop(object_cols, axis=1)
    X = pd.concat([numerical_X, X_encoded], axis=1)

    # Encode target values
    y = encode_targets(y)

    # Flatten JSON structure in alive_players
    stats_cols = ['health', 'armor', 'has_helmet',
     'has_defuser', 'money', 'team', 'inventory']
    for col in stats_cols:
        X = pd.merge(X, encode_players_stats(X, col),
         left_on='snapshot_id', right_on='snapshot_id', how='left')
        X = X.drop([f'player_unknown_{col}'], axis=1, errors='ignore')
        
    # Drop alive_players since it has been flattened
    X = X.drop(['alive_players'], axis=1)

    # Handle NaN's for groups: health, armor, has_helmet, has_defuser, money
    pc_range = range(1, 11)
    col_reg = ['health', 'armor', 'has_helmet', 'has_defuser', 'money']
    for col in col_reg:
        for pc in pc_range:
            X[f'player_{pc}_{col}'].fillna(0, inplace=True)
    
    # Handle dead players for groups: team
    for pc in pc_range:
        X[f'player_{pc}_team'].fillna('Dead', inplace=True)

    # Encode as 0/1's for groups: has_helmet, has_defuser
    col_labels = ['has_helmet', 'has_defuser']
    for col in col_labels:
        for pc in pc_range:
            X[f'player_{pc}_{col}'] = X[f'player_{pc}_{col}'] \
            .replace({True: 1, False: 0})
            
    # OH encode player teams
    player_cols = ['player_' + str(pc) + '_team' for pc in pc_range]
    X_encoded = encode_inputs(X, player_cols)
    numerical_X = X.drop(player_cols, axis=1)
    X = pd.concat([numerical_X, X_encoded], axis=1)

    # Create func to drop suffix
    pd.core.frame.DataFrame.drop_suffix = drop_suffix

    # Handle NaN in player's inventory
    for pc in pc_range:
        col = 'player_' + str(pc) + '_inventory'
        X = pd.merge(X, handle_nan_inventory(X[col]),
         left_index=True, right_index=True)
        X = X.drop([col + '_y'], axis=1)
        X = X.drop_suffix('_x')

    # Handle a case where cells occur as whitespace
    for pc in pc_range:
        col = 'player_' + str(pc) + '_inventory'
        text_empty_X = X[col].str.len() < 1
        indices_X = X.loc[text_empty_X].index
        for idx in indices_X:
            X.at[idx, col] = [{'item_type': 'None',
             'clip_ammo': 0, 'reserve_ammo': 0}]

    # Encode player's inventory
    for pc in pc_range:
        X = pd.merge(X, encode_players_inventory(X, pc),
         left_index=True, right_index=True)
        X = X.drop([f'player_{str(pc)}_inventory'], axis=1)

    # Handle NaN's for weapons and grenades
    X = X.fillna('None')

    # Do OH encoding of weapons and grenades
    import itertools
    col_inv = ['weapon_1', 'weapon_2', 'grenade_1',
     'grenade_2', 'grenade_3', 'grenade_4', 'grenade_5']
    player_cols = [['player_' + str(pc) + '_' + str(ci) for ci in col_inv]
     for pc in pc_range]
    player_cols = pd.Series(itertools.chain(*player_cols))
    X_encoded = encode_inputs(X, player_cols)
    numerical_X = X.drop(player_cols, axis=1)
    X = pd.concat([numerical_X, X_encoded], axis=1)

    # Remove duplicated columns
    X = X.loc[:,~X.columns.duplicated()]

    # Delete some leftover columns and drop grenades
    cols_leftover = '7_x|8_x|7_y|8_y|'
    cols_grenade = 'Grenade|grenade|Flashbang|C4|'
    cols_misc = 'None|Dead|Zeus|snapshot_id'
    all_cols = cols_leftover + cols_grenade + cols_misc
    X_encoded = X.drop(X.columns[X.columns.str.contains(all_cols)], axis=1)
    if '7' in X.columns: del X['7']

    # Cut off some decimals from round time
    X['round_status_time_left'] = X['round_status_time_left'].apply(
        lambda x: np.around(x, decimals=2))

    # Sum the stats of the teams
    X['ct_health'] = sum_player_cols(X, 'health', 'CT')
    X['t_health'] = sum_player_cols(X, 'health', 'Terrorist')
    X['ct_armor'] = sum_player_cols(X, 'armor', 'CT')
    X['t_armor'] = sum_player_cols(X, 'armor', 'Terrorist')
    X['ct_money'] = sum_player_cols(X, 'money', 'CT')
    X['t_money'] = sum_player_cols(X, 'money', 'Terrorist')
    X['ct_helmets'] = sum_player_cols(X, 'has_helmet', 'CT')
    X['t_helmets'] = sum_player_cols(X, 'has_helmet', 'Terrorist')
    X['ct_defuse_kits'] = sum_player_cols(X, 'has_defuser', 'CT')
    X['ct_players'] = sum([X[str(f'player_{i}_team_CT')] for i in pc_range])
    X['t_players'] = sum([X[str(f'player_{i}_team_Terrorist')] for i in pc_range])

    # Sum the weapons of the teams
    for weapon in cfg.weapon_list:
        X[f'ct_weapon_{weapon}'] = sum_player_cols(X, f'weapon_1_{weapon}', 'CT') \
         + sum_player_cols(X, f'weapon_2_{weapon}', 'CT')
        X[f't_weapon_{weapon}'] = sum_player_cols(X, f'weapon_1_{weapon}', 'Terrorist') \
         + sum_player_cols(X, f'weapon_2_{weapon}', 'Terrorist')
        
    # Drop individual player columns
    X = X.drop(X.columns[X.columns.str.contains('player_')], axis=1)

    # Save encoded results
    fw.write_csv(X, cfg.PROCESSED_DATA_DIR, 'X_full.csv')
    fw.write_csv(pd.Series(y), cfg.PROCESSED_DATA_DIR, 'y_full.csv')

def assign_scores(df):
    df = df.assign(ct_score=(df['current_score'].str[0]).astype(int))
    df = df.assign(t_score=(df['current_score'].str[1]).astype(int))
    return df

def encode_inputs(X, object_cols):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_encoded = pd.DataFrame(ohe.fit_transform(X[object_cols]))
    X_encoded.columns = ohe.get_feature_names(object_cols)
    X_encoded.index = X.index
    return X_encoded

def encode_targets(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    y_encoded = encoder.transform(y)
    return y_encoded

def encode_players_stats(df, col):
    df = df.set_index(['snapshot_id'])['alive_players']
    alive_players_list = [pd.DataFrame(alive_player) for alive_player in df]
    
    df = (pd.concat(alive_players_list, axis=1, keys=df.index)
          .transpose()
          .reset_index(level=[0,1])
          .rename(columns={'level_1':'player_stats'}))
    df = df.rename(columns={0:'player_1', 1:'player_2', 2:'player_3',
     3:'player_4', 4:'player_5',
                  5:'player_6', 6:'player_7', 7:'player_8',
                   8:'player_9', 9:'player_10', 10:'player_unknown'})
    
    df = df[df.player_stats == col].reset_index(drop=True)
    df = df.drop(['player_stats'], axis=1)
    df.columns = ['{}{}'.format(c, '' if c in ['snapshot_id']
     else '_' + str(col)) for c in df.columns]
    return df

def encode_players_inventory(df, pc):
    col = 'player_' + str(pc) + '_inventory'
    inv_list = [pd.DataFrame(inv) for inv in df[col]]
    pc_df = pd.concat(inv_list, axis=1, keys=df.index).transpose() \
    .reset_index(level=[1]).loc[lambda df: df['level_1'] == 'item_type'].rename(
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

def sum_player_cols(df, name, team, numPlayers=10):
    return sum([df[str(f'player_{i}_' + name)].where(
        df[str(f'player_{i}_team_' + team)] == 1.0, 0)
     for i in range(1, numPlayers + 1) if str(f'player_{i}_' + name) in df.columns])

if __name__ == '__main__':
    main()