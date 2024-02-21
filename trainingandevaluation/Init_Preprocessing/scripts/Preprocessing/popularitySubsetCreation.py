# -*- coding: utf-8 -*-

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import main


abspath = os.path.abspath(__file__)


def create_track_subset(dataPath = main.reducedDataPath, savePath = main.trackPopularityDataPath):
    """
    Computes a subset of the track popularity dataset based on the reduced dataset.
    The track popularity dataset includes the number of interactions (popularity) for each track and assigns
    each track to one of three categories: head, mid, or tail.
    """
    
    # Load the created dataset
    print("Loading the created dataset...")
    df = pd.read_csv(dataPath, sep='\t')
    print("Dataset loaded successfully.")
    
    # Calculate the number of users per track
    print("Calculating the number of users per track...")
    track_popularity = df.groupby('track_id')['user_id'].nunique().reset_index()
    track_popularity.columns = ['track_id', 'interactions']
    print("Interactions calculated.")
    
    # Sort the tracks based on the number of interactions
    print("Sorting the tracks based on the number of interactions...")
    track_popularity = track_popularity.sort_values('interactions', ascending=False).reset_index(drop=True)
    print("Tracks sorted.")
    
    # Calculate the total number of interactions
    print("Calculating the total number of interactions...")
    total_interactions = track_popularity['interactions'].sum()
    print("Total interactions calculated.")
    
    # Calculate the number of interactions corresponding to 20% of the total interactions
    print("Calculating the number of interactions corresponding to 20% of the total interactions...")
    interaction_cutoff = total_interactions * 0.2
    print("Interaction cutoff calculated.")
    
    # Find the index where the cumulative sum exceeds the interaction cutoff
    print("Finding the index where the cumulative sum exceeds the interaction cutoff...")
    head_index = track_popularity['interactions'].cumsum().ge(interaction_cutoff).idxmax()
    print("Head index found.")    
    
    # Find the index where the cumulative sum reaches or exceeds the interaction cutoff
    print("Finding the index where the cumulative sum reaches or exceeds the interaction cutoff...")
    tail_index = track_popularity['interactions'].cumsum().ge(total_interactions - interaction_cutoff).idxmax()
    print("Tail index found.")

    def assign_popularity(row):
        
        if row.name <= head_index:
            return 'head'
        elif row.name >= tail_index:
            return 'tail'
        else:
            return 'mid'
        
    print("Assigning popularity categories to tracks...")
    track_popularity['popularity'] = track_popularity.apply(assign_popularity, axis=1)
    print("Popularity categories assigned.")
    
    
    print("Saving the dataset to", savePath)
    track_popularity.to_csv(savePath, index=False)
    print("Dataset saved.")
    
    print('Dataset shape: ' )
    print(track_popularity.shape)

