# -*- coding: utf-8 -*-

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import main


abspath = os.path.abspath(__file__)

def create_profile_subset(dataPath = main.reducedDataPath, savePath = main.userProfileDataPath, trackPopularityPath=main.trackPopularityDataPath):
    """
    Computes information about the user profiles based on the listening counts dataset 
    and the track popularity dataset.
    Can be used for further analysis of the user profiles.
    """
    
    print("Loading the created dataset...")
    df = pd.read_csv(dataPath, sep='\t')
    print("Dataset loaded successfully.")
    print("Loading track popularity data...")
    track_popularity = pd.read_csv(trackPopularityPath)
    print("Track popularity data loaded successfully.")
    
    print("Merging datasets...")
    df = pd.merge(df, track_popularity, on='track_id')
    print("Datasets merged.")
    
    print("Calculating the number of interactions per user...")
    user_interactions = df.groupby('user_id')['track_id'].nunique().reset_index()
    user_interactions.columns = ['user_id', 'interactions']
    print("Interactions calculated.")
    
    print("Calculating the number of head, mid, and tail items for each user...")
    user_items = df.groupby(['user_id', 'popularity'])['track_id'].nunique().reset_index()
    user_items.columns = ['user_id', 'popularity', 'items']
    print("Items calculated.")
    
    print("Calculating the total number of items for each user...")
    total_items_per_user = user_items.groupby('user_id')['items'].sum().reset_index()
    total_items_per_user.columns = ['user_id', 'total_items']
    print("Total items calculated.")
    
    print("Merging the total_items_per_user DataFrame with user_items...")
    user_items = pd.merge(user_items, total_items_per_user, on='user_id')
    print("DataFrames merged.")
    
    print("Calculating the ratio of head, mid, and tail items for each user...")
    user_items['ratio'] = user_items['items'] / user_items['total_items']
    print("Ratio calculated.")
    
    print("Sorting the user_items dataframe based on user_id...")
    user_items = user_items.sort_values('user_id').reset_index(drop=True)
    print("Dataframe sorted.")
    
    print("Calculating the ratio of head items for each user...")
    head_ratio = user_items[user_items['popularity'] == 'head'][['user_id', 'ratio']]
    head_ratio.columns = ['user_id', 'head_ratio']
    print("Head ratio calculated.")
    
    print("Calculating the ratio of mid items for each user...")
    mid_ratio = user_items[user_items['popularity'] == 'mid'][['user_id', 'ratio']]
    mid_ratio.columns = ['user_id', 'mid_ratio']
    print("Mid ratio calculated.")
    
    print("Calculating the ratio of tail items for each user...")
    tail_ratio = user_items[user_items['popularity'] == 'tail'][['user_id', 'ratio']]
    tail_ratio.columns = ['user_id', 'tail_ratio']
    print("Tail ratio calculated.")
    
    print("Merging the head_ratio, mid_ratio, and tail_ratio dataframes...")
    user_types = pd.merge(head_ratio, mid_ratio, on='user_id', how='outer')
    user_types = pd.merge(user_types, tail_ratio, on='user_id', how='outer')
    user_types = pd.merge(user_types, total_items_per_user, on='user_id', how='outer')
    print("Dataframes merged.")
    
    print("Filling missing values with 0...")
    user_types[['head_ratio', 'mid_ratio', 'tail_ratio']] = user_types[['head_ratio', 'mid_ratio', 'tail_ratio']].fillna(0)
    print("Missing values filled.")
    
    print("Sorting the user_types dataframe based on head_ratio in descending order...")
    user_types = user_types.sort_values('head_ratio', ascending=False).reset_index(drop=True)
    print("Dataframe sorted.")
    
    print("Calculating the number of users corresponding to 20% of the total users...")
    user_count = user_types.shape[0]
    blockbuster_count = int(user_count * 0.2)
    niche_count = int(user_count * 0.2)
    print("Counts calculated.")
    
    print("Assigning 'user_type' based on ranking...")
    user_types.loc[:blockbuster_count, 'user_type'] = 'Blockbuster-focused'
    user_types.loc[user_count - niche_count:, 'user_type'] = 'Niche'
    user_types.loc[blockbuster_count+1:user_count - niche_count-1, 'user_type'] = 'Diverse'
    print("User types assigned.")
    
    print("Saving the dataset to", savePath)
    user_types.to_csv(savePath, index=False)

    print('Dataset shape: ' )
    print(user_types.shape)
    print('Dataset saved to', savePath)
    
    
