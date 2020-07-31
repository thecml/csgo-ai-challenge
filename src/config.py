import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent.parent
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
RAW_DATA_DIR = Path.joinpath(ROOT_DIR, 'data/raw')
PROCESSED_DATA_DIR = Path.joinpath(ROOT_DIR, 'data/processed')
INTERIM_DATA_DIR = Path.joinpath(ROOT_DIR, 'data/interim')
EXTERNAL_DATA_DIR = Path.joinpath(ROOT_DIR, 'data/external')