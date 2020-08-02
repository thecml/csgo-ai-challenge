import json
import os
from pathlib import Path

weapon_list = ['Ak47', 'Aug', 'Awp', 'Bizon', 'Cz75Auto', 'Elite',
 'Famas', 'G3sg1', 'GalilAr', 'Glock', 'M249', 'M4a1S', 'M4a4', 'Mac10',
 'Mag7', 'Mp5sd', 'Mp7', 'Mp9', 'Negev', 'Nova', 'P90', 'R8Revolver', 'Sawedoff',
 'Scar20', 'Sg553', 'Ssg08', 'Ump45', 'Xm1014', 'Deagle', 'FiveSeven', 'Glock', 'UspS',
 'P250', 'P2000', 'Tec9', 'Cz75Auto']

ROOT_DIR = Path(__file__).absolute().parent.parent
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
LOGS_DIR = Path.joinpath(ROOT_DIR, 'logs')
RAW_DATA_DIR = Path.joinpath(ROOT_DIR, 'data/raw')
PROCESSED_DATA_DIR = Path.joinpath(ROOT_DIR, 'data/processed')
INTERIM_DATA_DIR = Path.joinpath(ROOT_DIR, 'data/interim')
EXTERNAL_DATA_DIR = Path.joinpath(ROOT_DIR, 'data/external')