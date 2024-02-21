# -*- coding: utf-8 -*-

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import main

abspath = os.path.abspath(__file__)


def create_subset(savePath = main.reducedDataPath, dataPath=main.fullDataPath, spotifyPath= main.fullSpotifyURIPath):
    """
    Preprocessing of the original dataset
    
    Creates a subset of the dataset by filtering out interactions with playcount smaller than 2
    and tracks not present in the Spotify dataset.
    Also removes iteratively tracks listened to by fewer than 5 different users and 
    users who listened to fewer than 5 different tracks.
    
    """
    
    # get track ids from songs in the spotify dataset
    uris = pd.read_csv(spotifyPath)
    uri_track_ids = list(uris['track_id'])
    del uris
    
    
    # Define the batch size and initialize an empty DataFrame to store the results
    batch_size = 100000
    
    i = 0
    
    saved_valid_tracks = set()
    saved_valid_users = set()
    
    # Create an empty dataframe to store the interactions that are not valid yet but might become valid after processing more interactions
    df = pd.DataFrame(columns=['user_id', 'track_id', 'count'])
    
    # Create a file to save the reduced dataset
    with open(savePath, 'w') as f:
        f.write('user_id\ttrack_id\tcount\n')  # Write header to the file


        print('Load batches...')
        # Iterate over the raw dataset in chunks
        for i, chunk in enumerate(pd.read_csv(dataPath, delimiter='\\', chunksize=batch_size)):
            if i % 100 == 0:
                print('it:' + str(i * batch_size) + '; df size: ' + str(len(df)))
        
            # Split the merged column into separate columns
            chunk[['user_id', 'track_id', 'count']] = chunk['user_id\ttrack_id\tcount'].str.split('\t', expand=True)
            
            # Drop the merged column
            chunk.drop('user_id\ttrack_id\tcount', axis=1, inplace=True)
        
            # Process each chunk
            chunk['count'] = pd.to_numeric(chunk['count'], downcast='integer')
            chunk['track_id'] = pd.to_numeric(chunk['track_id'], downcast='integer')
            
            # Filter out interactions where the playcount is smaller than 2
            chunk = chunk[chunk['count'] >= 2]
            chunk = chunk[chunk['track_id'].isin(uri_track_ids)]
            
            # if the item and the user are valid before -> save the interaction
            valid_interactions = chunk[(chunk['user_id'].isin(saved_valid_users)) & (chunk['track_id'].isin(saved_valid_tracks))]

            if not valid_interactions.empty:
                valid_interactions.to_csv(f, sep='\t', index=False, header=False, mode='a')
                chunk = chunk[~chunk.index.isin(valid_interactions.index)]
                
            # if not, compute whether the interaction is valid
            chunk, saved_valid_tracks, saved_valid_users, df_to_save = compute_valid_interactions(chunk, saved_valid_tracks, saved_valid_users)
            
            if not df_to_save.empty:
                df_to_save.to_csv(f, sep='\t', index=False, header=False, mode='a')
                
            # Append the processed chunk containing of items that are not valid yet to the main dataframe
            df = pd.concat([df, chunk], ignore_index=True)
            
            # Clear the chunk variable to free memory
            del chunk
    
            i += 1
            
            # After processing 100 chunks, test whether some of the interactions are valid now
            if i % 100 == 0:
                df, saved_valid_tracks, saved_valid_users, df_to_save = compute_valid_interactions(df, saved_valid_tracks, saved_valid_users)
                
                if not df_to_save.empty:
                    df_to_save.to_csv(f, sep='\t', index=False, header=False, mode='a')
        # Check whether some of the remaining interactions are valid
        df, saved_valid_tracks, saved_valid_users, df_to_save = compute_valid_interactions(df, saved_valid_tracks, saved_valid_users)
        
        # Save the valid interactions
        if not df_to_save.empty:
            df_to_save.to_csv(f, sep='\t', index=False, header=False, mode='a')
            
    print('Dataset saved to ' + savePath)
    
    
    
    
    
def compute_valid_interactions(df, saved_valid_tracks, saved_valid_users):
    # Calculate the count of unique users per track
    track_user_count = df.groupby('track_id')['user_id'].nunique().reset_index()
    
    # Calculate the count of unique tracks per user
    user_track_count = df.groupby('user_id')['track_id'].nunique().reset_index()
    valid_tracks = track_user_count['track_id'].tolist()
    valid_users = user_track_count['user_id'].tolist()
    
    
    sub_df = df.copy(deep = True)
    
    
    while True:
        # Filter out tracks listened to by fewer than 5 different users
        new_valid_tracks = track_user_count[track_user_count['user_id'] >= 5]['track_id'].tolist()
        sub_df = sub_df[sub_df['track_id'].isin(new_valid_tracks) | sub_df['track_id'].isin(saved_valid_tracks)]
        
        # Filter out users who listened to fewer than 5 different tracks
        new_valid_users = user_track_count[user_track_count['track_id'] >= 5]['user_id'].tolist()
        sub_df = sub_df[sub_df['user_id'].isin(new_valid_users) | sub_df['user_id'].isin(saved_valid_users)]
        
        # If the number of valid tracks or users has decreased, update the valid tracks and users and recalculate the counts
        if len(new_valid_tracks) < len(valid_tracks) or len(new_valid_users) < len(valid_users):
            valid_tracks = new_valid_tracks
            valid_users = new_valid_users
            
            # Calculate the count of unique users per track
            track_user_count = sub_df.groupby('track_id')['user_id'].nunique().reset_index()
            
            # Calculate the count of unique tracks per user
            user_track_count = sub_df.groupby('user_id')['track_id'].nunique().reset_index()
            
            
        # If valid tracks were found, save the valid interactions and remove them from the main dataframe    
        elif not sub_df.empty:  
            saved_valid_tracks.update(sub_df['track_id'])
            saved_valid_users.update(sub_df['user_id'])
            df = df[~df.index.isin(sub_df.index)]
            break
        else:
            break
    return df, saved_valid_tracks, saved_valid_users, sub_df
