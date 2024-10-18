import os

# Base directory for storing data and models
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Directory paths for data storage and caching
DATA_DIR_STORAGE = os.path.join(BASE_DIR, "data")
DATA_DIR_MODELS = os.path.join(BASE_DIR, "models")
DATA_DIR_EVAL = os.path.join(BASE_DIR, "eval")
DATA_DIR_CACHE = os.path.join(BASE_DIR, "cache")
