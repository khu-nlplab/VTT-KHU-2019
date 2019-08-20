import tensorflow as tf
import numpy as np
import json
import os

_JSON_TAG = 'json'

def load_glove(path):

    word2vec = {}

    with open(path) as glove_file:
        for line in glove_file:
            vals = line.split()
            word = str(''.join(vals[:-300]))
            vector = np.asarray(vals[-300:], dtype='float32')

            word2vec[word] = vector

    return word2vec


def make_dir(path):
    if not tf.gfile.Exists(path):
        tf.logging.info("creating directory %s" % path)
        tf.gfile.MakeDirs(path)

def read_file(dataset_path, file_type = None):

    with open(dataset_path) as file:
        if file_type == 'json':
            raw_data = json.load(file)
        else:
            raw_data = file.readlines()


    return raw_data

def write_file(path, content, file_type=None):

    if file_type == 'json':
        content = json.dumps(content)

    with tf.gfile.Open(path, mode="w+") as file:
        file.write(str(content))

def find_file(dir_path, filename, max_depth=4):

    for root, dirs, files in os.walk(dir_path):
        if filename in files:
            return os.path.join(root, filename)

        depth = root[len(dir_path) + 1:].count(os.sep)
        if depth > max_depth:
            del dirs[:]

    return None

def validate_path(path):
    if not tf.gfile.Exists(path):
        return False

    return True

def get_spilited_data(filepaths):

    train = read_file(filepaths[0], _JSON_TAG)
    validation = read_file(filepaths[1], _JSON_TAG)
    test = read_file(filepaths[2], _JSON_TAG)

    return train, validation, test