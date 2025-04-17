import os
from glob import glob
import json


def get_recursive_files(dir):
    files = glob(dir + "/**/*", recursive=True)
    return [os.path.basename(x) for x in files if os.path.isfile(x)]


def dump_json(data, path):
    open_dir = os.path.dirname(path)
    if not os.path.exists(open_dir):
        os.makedirs(open_dir)
    json.dump(data, open(path, "w"))


def load_json(path):
    return json.load(open(path))
