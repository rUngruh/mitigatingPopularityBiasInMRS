# -*- coding: utf-8 -*-


import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import main
import pandas as pd
import numpy as np


def create_spotify_subset(dataPath = main.fullSpotifyURIPath, tracksPath = main.trackPopularityDataPath, savePath = main.spotifyTracksPath):
    """
    Computes a subset of the spotify uris dataset based on the track popularity dataset.
    Since track_popularities only includes valid items, the spotify uris dataset is filtered accordingly.
    """
    
    # Load the created dataset
    print("Loading the created dataset...")
    
    # Load the datasets
    df = pd.read_csv(dataPath, delimiter='\t')
    track_popularities = pd.read_csv(tracksPath, delimiter='\t')
    
    # Merge the datasets on 'track_id' and set 'uri' to np.nan for missing entries 
    # (should not be the case, but just in case)
    spotify_uris = pd.merge(track_popularities, df, on='track_id', how='left')
    spotify_uris['uri'] = spotify_uris['uri'].fillna(np.nan)
    
    # Delete all columns except 'track_id' and 'uri'
    spotify_uris = spotify_uris[['track_id', 'uri']]
    
    print("Dataset loaded successfully.")
    
    # Save the modified dataset
    print("Saving the dataset to", savePath)
    spotify_uris.to_csv(savePath, index=False)
    print("Dataset saved.")
