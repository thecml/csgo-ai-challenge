import pandas as pd
import numpy as np

maps = ['map_de_cache', 'map_de_dust2', 'map_de_inferno', 'map_de_mirage',
'map_de_nuke', 'map_de_overpass', 'map_de_train', 'map_de_vertigo']

def add_round_time(row):
    if row['round_status_FreezeTime'] == 1:
        return row['round_status_time_left'] + 155
    return row['round_status_time_left']

def normalize_maps(row):
    for col in maps:
        if row[col] == 1:
            return col.split("map_", 1)[1]

def normalize_bomb_planted(row):
    if row == 1.0:
        return True
    return False

def normalize_round_winner(row):
    if row == 0:
        return 'CT'
    return 'T'

X = pd.read_csv('C:\\Users\\cml\\Desktop\\csgo-ai-challenge\\data\\processed\\X_full.csv')
y = pd.read_csv('C:\\Users\\cml\\Desktop\\csgo-ai-challenge\\data\\processed\\y_full.csv')

df = pd.concat([X, y], axis=1)

df['round_status_time_left'] = df.apply(add_round_time, axis=1)
df = df.drop(['round_status_FreezeTime', 'round_status_Normal'], axis=1)
df['round_status_time_left'] = df['round_status_time_left'].clip(0, 175)
df = df.rename({'round_status_time_left': 'time_left', 'round_status_BombPlanted': 'bomb_planted'}, axis=1)

df['map'] = df.apply(normalize_maps, axis=1)
df = df.drop(columns=maps)

cols_to_order = ['time_left', 'ct_score', 't_score', 'map']
new_columns = cols_to_order + (df.columns.drop(cols_to_order).tolist())
df = df[new_columns]
df.columns = df.columns.str.lower()
df = df.rename({'ct_players':'ct_players_alive', 't_players':'t_players_alive'}, axis=1)

df['time_left'] = df['time_left'].apply(lambda x: np.around(x, 2))
df['bomb_planted'] = df['bomb_planted'].apply(normalize_bomb_planted)
df['round_winner'] = df['round_winner'].apply(normalize_round_winner)

df = df.drop(df.columns[df.columns.str.contains('c4')], axis=1)

df.to_csv('C:\\Users\\cml\\Desktop\\csgo-ai-challenge\\data\\processed\\csgo_round_snapshots.csv', index=False)
