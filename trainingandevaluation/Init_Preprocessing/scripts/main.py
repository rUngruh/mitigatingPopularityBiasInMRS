# -*- coding: utf-8 -*-



import os
import pandas as pd

abspath = os.path.abspath(__file__)

from scripts.Preprocessing import data_preprocessing as pre




# Data paths
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
completeDataPath = os.path.join(path, 'data')


fullSpotifyURIPath = os.path.join(path,'data', 'raw', 'spotify-uris.tsv.bz2')
fullDataPath = os.path.join(path,'data', 'raw', 'listening-counts.tsv.bz2')
reducedDataPath = os.path.join(completeDataPath, 'Processed', 'reduced_listening_counts.csv')
trackPopularityDataPath = os.path.join(completeDataPath, 'Processed', 'track_popularity.csv')
userProfileDataPath = os.path.join(completeDataPath, 'Processed', 'user_profiles.csv')
initAMatrixPath = os.path.join(completeDataPath, 'Matrices', 'init_A_matrix.npz')
initRMatrixPath = os.path.join(completeDataPath, 'Matrices', 'init_R_matrix.npz')
initSavePathMappings = os.path.join(completeDataPath, 'Matrices', 'init_mappings.npz')
savePathTest = os.path.join(completeDataPath, 'Matrices', 'A_test.npz')
savePathTrain = os.path.join(completeDataPath, 'Matrices', 'A_train.npz')
spotifyTracksPath = os.path.join(completeDataPath, 'Processed', 'spotify_uris.csv')



plotPath = os.path.join(path, 'plots')
evaluationPath = os.path.join(path, 'evaluation')


"""
Run preprocessing of data
Add the required files to the data folder

It has to be remarked that some of the preprocessing steps are time consuming and require a lot of memory.
Therefore, it is recommended to run some of the steps separately.
"""

# Read and filter data
import Preprocessing.subsetCreationSpotifyFilter as scsf

scsf.create_subset(savePath=reducedDataPath, dataPath=fullDataPath, spotifyPath=fullSpotifyURIPath)


#Create trackData
import Preprocessing.popularitySubsetCreation as psc

psc.create_track_subset(dataPath = reducedDataPath, savePath = trackPopularityDataPath)

# Recompute valid Spotify URIs
import Preprocessing.spotifySubsetCreation as ssc

ssc.create_spotify_subset(dataPath = fullSpotifyURIPath, tracksPath = trackPopularityDataPath, savePath = spotifyTracksPath)


# Create matrices
import Preprocessing.data_preprocessing as dp

dp.create_matrices(dataPath = reducedDataPath, savePathA = initAMatrixPath, savePathR = initRMatrixPath, savePathMappings = initSavePathMappings)
dp.create_train_test_split(matrixdataPath = initRMatrixPath, test_size=0.2, savePathTest = savePathTest,  savePathTrain = savePathTrain)


# Create user profiles
import Preprocessing.userProfileSubsetCreation as upsc

upsc.create_profile_subset(dataPath = reducedDataPath, savePath = userProfileDataPath, trackPopularityPath=trackPopularityDataPath)




# load matrices
test, train, mappings = pre.load_train_and_test_matrix(dataPathTest = savePathTest, dataPathTrain = savePathTrain, dataPathMappings = initSavePathMappings)
user_index_map, track_index_map = mappings



track_popularity = pd.read_csv(trackPopularityDataPath)
user_profiles = pd.read_csv(userProfileDataPath)


