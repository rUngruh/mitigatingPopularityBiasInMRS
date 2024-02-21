# -*- coding: utf-8 -*-

import os


abspath = os.path.abspath(__file__)

import numpy as np
from scipy.sparse import load_npz


def load_train_and_test_matrix(dataPathTest="", dataPathTrain="",
                               dataPathMappings=""):
    
    """
    load the train and test matrix from the given paths
    """
    
    print('Loading train and test set...')
    test = load_npz(dataPathTest)
    train = load_npz(dataPathTrain)
    mappings = np.load(dataPathMappings, allow_pickle=True)
    dec_mappings = [mappings['user_index_map_inv'].item(), mappings['track_index_map_inv'].item()]
    print('Train and test set loaded.')
    return train, test, dec_mappings

