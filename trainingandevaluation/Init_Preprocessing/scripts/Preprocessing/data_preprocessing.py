# -*- coding: utf-8 -*-


import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


abspath = os.path.abspath(__file__)


import pandas as pd
import main
import numpy as np
from scipy.sparse import lil_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split

def create_matrices(dataPath=main.reducedDataPath, savePathA=main.initAMatrixPath, savePathR=main.initRMatrixPath,
                    savePathMappings=main.initSavePathMappings):
    """
    Create the matrices derived from the reduced data set
    """
    
    # Load the created dataset
    print("Loading the created dataset...")
    df = pd.read_csv(dataPath, sep='\t')
    print("Dataset loaded successfully.")

    # Generate unique users and tracks
    print("Generating unique users and tracks...")
    unique_users = df['user_id'].unique()
    unique_tracks = df['track_id'].unique()

    num_users = len(unique_users)
    num_tracks = len(unique_tracks)

    # Maps user and track IDs to indices corresponding to the rows and columns of the matrices
    print("Creating user and track index mappings...")
    user_index_map = {user_id: index for index, user_id in enumerate(unique_users)}
    track_index_map = {track_id: index for index, track_id in enumerate(unique_tracks)}

    # Initialize empty matrices A and R
    print("Initializing matrices A and R...")
    A = lil_matrix((num_users, num_tracks), dtype=np.int8)
    R = lil_matrix((num_users, num_tracks), dtype=np.int8)

    # Add rows iteratively to the matrices A and R
    # A is the count matrix and R is the binary implicit feedback matrix
    print("Populating matrices A and R...")
    num_rows=len(df)
    for i, row in df.iterrows():
        if i %1000000 == 0:
            print(f'Progress: {((i/num_rows)*100):.4f}%')
        user_id = row['user_id']
        track_id = row['track_id']
        count = row['count']

        user_index = user_index_map[user_id]
        track_index = track_index_map[track_id]

        A[user_index, track_index] = count
        R[user_index, track_index] = 1 if count > 0 else 0

    # Create inverse mappings
    user_index_map_inv = {index: user_id for user_id, index in user_index_map.items()}
    track_index_map_inv = {index: track_id for track_id, index in track_index_map.items()}

    print("Saving matrices A and R to disk...")
    # Save matrices A and R to disk
    save_npz(savePathA, A.tocsr())
    save_npz(savePathR, R.tocsr())

    # Save mappings
    np.savez(savePathMappings, user_index_map_inv=user_index_map_inv, track_index_map_inv=track_index_map_inv)
    print("Matrices and mappings saved.")

    # Return matrices and inverse mappings
    return A, R, user_index_map_inv, track_index_map_inv




def create_train_test_split(matrixdataPath=main.initRMatrixPath, test_size=0.2, savePathTest=main.savePathTest,
                            savePathTrain=main.savePathTrain):
    """
    Split the user-item matrix into training and test sets and save them
    test_size: The proportion of the test set
    """
    user_item_matrix =load_npz(matrixdataPath).tolil()

    # Get the indices of non-zero entries in the user-item matrix
    nonzero_indices = user_item_matrix.nonzero()

    # Split the non-zero indices into training and test sets
    train_indices, test_indices = train_test_split(np.array(nonzero_indices).T, test_size=test_size)

    # Create the train and test matrices with zero entries
    train_matrix = lil_matrix(user_item_matrix.shape, dtype=np.int8)
    test_matrix = lil_matrix(user_item_matrix.shape, dtype=np.int8)

    # Assign the non-zero entries to the train and test matrices
    train_matrix[train_indices[:, 0], train_indices[:, 1]] = user_item_matrix[train_indices[:, 0], train_indices[:, 1]]
    test_matrix[test_indices[:, 0], test_indices[:, 1]] = user_item_matrix[test_indices[:, 0], test_indices[:, 1]]

    save_npz(savePathTrain, train_matrix.tocsr())
    save_npz(savePathTest, test_matrix.tocsr())

    print(user_item_matrix.shape, train_matrix.shape, test_matrix.shape)


def load_train_and_test_matrix(dataPathTest=main.savePathTest, dataPathTrain=main.savePathTrain,
                               dataPathMappings=main.initSavePathMappings):
    """
    Loads the saved train and test matrices and the mappings
    """
    print('Loading train and test set...')
    test = load_npz(dataPathTest)
    train = load_npz(dataPathTrain)
    mappings = np.load(dataPathMappings, allow_pickle=True)
    dec_mappings = [mappings['user_index_map_inv'].item(), mappings['track_index_map_inv'].item()]
    print('Train and test set loaded.')
    return train, test, dec_mappings


