# -*- coding: utf-8 -*-

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
abspath = os.path.abspath(__file__)


import main
import pandas as pd
import numpy as np

def analyse_dataset(interactionDataPath = main.reducedDataPath):
    """
    Analyses the dataset and prints the number of interactions, unique users, and unique tracks
    """
    # Load the created dataset
    print("Loading the dataset...")
    df = pd.read_csv(interactionDataPath, sep='\t')
    print("Dataset loaded successfully.")
    print(len(df))
    # Calculate the number of interactions
    interactions = len(df)
    
    # Calculate the number of unique user_ids
    users = df['user_id'].nunique()
    
    # Calculate the number of unique track_ids
    tracks = df['track_id'].nunique()
    
    # Print the results
    print("Interactions:", interactions)
    print("Users:", users)
    print("Tracks:", tracks)
    

from Preprocessing.data_preprocessing import load_matrices

def analyse_sparsity(dataPathA = main.initAMatrixPath, dataPathR = main.initRMatrixPath, dataPathMappings = main.initSavePathMappings):
    """
    Analyses the sparsity of the matrix R
    """
    
    print("Loading the matrices...")
    A, R, mappings = load_matrices(dataPathA, dataPathR, dataPathMappings)
    print("Matrices loaded successfully.")
    # Calculate the total number of elements in R
    total_elements = R.shape[0] * R.shape[1]
    
    # Calculate the number of non-zero elements in R
    entries = np.sum(R != 0)
    
    
    # Calculate the sparsity of R
    sparsity = (1 - (entries / total_elements)) * 100
    
    # Print the sparsity
    print("Sparsity of matrix R: {:.4f}%".format(sparsity))

        
analyse_dataset(interactionDataPath = main.SpreducedDataPath)   
     
analyse_sparsity(dataPathA = main.SpinitAMatrixPath, dataPathR = main.SpinitRMatrixPath, dataPathMappings = main.SpinitSavePathMappings)
    







