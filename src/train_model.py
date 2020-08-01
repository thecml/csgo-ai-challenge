import numpy as np
import pandas as pd
import src.file_reader as fr
import src.file_writer as fw
import src.config as cfg
from tensorflow import keras

def main():
    nrows = 100
    X_train = fr.read_csv(cfg.INTERIM_DATA_DIR, 'X_train.csv', nrows=nrows)
    X_valid = fr.read_csv(cfg.INTERIM_DATA_DIR, 'X_valid.csv', nrows=nrows)
    y_train = fr.read_csv(cfg.INTERIM_DATA_DIR, 'y_train.csv', nrows=nrows)
    y_valid = fr.read_csv(cfg.INTERIM_DATA_DIR, 'y_valid.csv', nrows=nrows)

    X_train = prepare_data(X_train)
    X_valid = prepare_data(X_valid)

    fw.write_csv(X_train, cfg.PROCESSED_DATA_DIR, 'X_train.csv')
    fw.write_csv(X_valid, cfg.PROCESSED_DATA_DIR, 'X_valid.csv')
    fw.write_csv(y_train, cfg.PROCESSED_DATA_DIR, 'y_train.csv')
    fw.write_csv(y_valid, cfg.PROCESSED_DATA_DIR, 'y_valid.csv')

def make_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def prepare_data(df):
    cols_leftover = '7_x|8_x|7_y|8_y|'
    cols_grenade = 'Grenade|grenade|Flashbang|C4|'
    # cols_pistol = 'Deagle|FiveSeven|Glock|UspS|P250|P2000|Tec9|Cz75Auto'
    cols_misc = 'None|Dead|Zeus|snapshot_id'
    all_cols = cols_leftover + cols_grenade + cols_misc
    df = df.drop(df.columns[df.columns.str.contains(all_cols)], axis=1)

    # Delete another leftover column
    if '7' in df.columns: del df['7']

    # Cut off some decimals from round time
    df['round_status_time_left'] = df['round_status_time_left'].apply(
        lambda x: np.around(x, decimals=2))

    # Sum all the health of the players
    # TODO: How do we check that the player plays on CT/T team?
    # Each player has a player_1_team_CT and player_1_team_Terrorist column
    numPlayers = 10
    df['ct_health'] = sumPlayerCols(df, 'health', 'CT')
    df['t_health'] = sumPlayerCols(df, 'health', 'Terrorist')
    df['ct_armor'] = sumPlayerCols(df, 'armor', 'CT')
    df['t_armor'] = sumPlayerCols(df, 'armor', 'Terrorist')
    df['ct_money'] = sumPlayerCols(df, 'money', 'CT')
    df['t_money'] = sumPlayerCols(df, 'money', 'Terrorist')
    df['ct_helmets'] = sumPlayerCols(df, 'has_helmet', 'CT')
    df['t_helmets'] = sumPlayerCols(df, 'has_helmet', 'Terrorist')
    df['ct_defuse_kits'] = sumPlayerCols(df, 'has_defuser', 'CT')
    df['ct_players'] = sum([df[str(f'player_{i}_team_CT')] for i in range(1, numPlayers + 1)])
    df['t_players'] = sum([df[str(f'player_{i}_team_Terrorist')] for i in range(1, numPlayers + 1)])

    # Weapons
    weapon_list = ['Ak47', 'Aug', 'Awp', 'Bizon', 'Cz75Auto', 'Elite', 'Famas', 'G3sg1', 'GalilAr', 'Glock', 'M249', 'M4a1S', 'M4a4', 'Mac10', 'Mag7', 'Mp5sd', 'Mp7', 'Mp9', 'Negev', 'Nova', 'P90', 'R8Revolver', 'Sawedoff', 'Scar20', 'Sg553', 'Ssg08', 'Ump45', 'Xm1014', 'Deagle', 'FiveSeven', 'Glock', 'UspS', 'P250', 'P2000', 'Tec9', 'Cz75Auto']

    for weapon in weapon_list:
        df[f'ct_weapon_{weapon}'] = sumPlayerCols(df, f'weapon_1_{weapon}', 'CT') + sumPlayerCols(df, f'weapon_2_{weapon}', 'CT')
        df[f't_weapon_{weapon}'] = sumPlayerCols(df, f'weapon_1_{weapon}', 'Terrorist') + sumPlayerCols(df, f'weapon_2_{weapon}', 'Terrorist')

    # player_health_cols = [f'player_{i}_health' for i in range(1, numPlayers + 1)]
    # player_armor_cols = [f'player_{i}_armor' for i in range(1, numPlayers + 1)]
    # player_money_cols = [f'player_{i}_money' for i in range(1, numPlayers + 1)]
    # player_helmet_cols = [f'player_{i}_has_helmet' for i in range(1, numPlayers + 1)]
    # player_defuse_kits_cols = [f'player_{i}_has_defuser' for i in range(1, numPlayers + 1)]
    # player_team_ct_cols = [f'player_{i}_team_CT' for i in range(1, numPlayers + 1)]
    # player_team_terrorist_cols = [f'player_{i}_team_Terrorist' for i in range(1, numPlayers + 1)]
    # player_ct_weapons_cols = [f'player_{i}_team_Terrorist' for i in range(1, numPlayers + 1)]
    df = df.drop(df.columns[df.columns.str.contains('player_')], axis=1)
    
    # df = df.drop(
    #     player_health_cols +
    #     player_armor_cols +
    #     player_money_cols +
    #     player_helmet_cols +
    #     player_defuse_kits_cols +
    #     player_team_ct_cols +
    #     player_team_terrorist_cols,
    #     axis=1)
    return df

def sumPlayerCols(df, name, team, numPlayers=10):
    return sum([df[str(f'player_{i}_' + name)].where(df[str(f'player_{i}_team_' + team)] == 1.0, 0) for i in range(1, numPlayers + 1) if str(f'player_{i}_' + name) in df.columns])

if __name__ == '__main__':
    main()