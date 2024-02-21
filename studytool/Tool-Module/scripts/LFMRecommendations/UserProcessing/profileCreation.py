# -*- coding: utf-8 -*-

import pandas as pd

def create_user_profile(user_id, track_ids, track_popularities, num_interactions):
    # filter the track_popularities
    t_pops = track_popularities.copy()
    t_pops = t_pops[t_pops['track_id'].isin(track_ids)]
    
    # calculate the ratios of head, mid and tail interactions
    num_filtered_interactions = t_pops.shape[0]
    pop_counts = t_pops['popularity'].value_counts()
    head_ratio = pop_counts.get('head', 0) / num_filtered_interactions
    mid_ratio = pop_counts.get('mid', 0) / num_filtered_interactions
    tail_ratio = pop_counts.get('tail', 0) / num_filtered_interactions
    
    # calculate mean and median popularity
    mean_interactions = t_pops['interactions'].mean()
    median_interactions = t_pops['interactions'].median()
    
    # Classify the user into one of the three categories
    if head_ratio > 0.430726:
        user_type = "Blockbuster_focused"
    elif head_ratio < 0.133394:
        user_type = "Niche"
    else:
        user_type = "Diverse"
        
    # Save the user profile as a dataframe
    user_profile = pd.DataFrame({
        'user_id': [user_id],
        'num_interactions':[num_interactions],
        'num_filtered_interactions':[num_filtered_interactions],
        'head_ratio':[head_ratio],
        'mid_ratio':[mid_ratio],
        'tail_ratio':[tail_ratio],
        'user_type':[user_type],
        'mean_interactions':[mean_interactions],
        'median_interactions':[median_interactions]
        })
    return user_profile
