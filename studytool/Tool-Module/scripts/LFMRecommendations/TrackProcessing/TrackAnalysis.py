# -*- coding: utf-8 -*-

import math
import pandas as pd
import os
from datetime import datetime

def analyse_recommendations(user_id, base, fair, cp, user_profile, track_popularities):
    """
    Compute statistics of the recommendation lists and save them in a dataframe
    """
    # Base metrics
    profile_mean_interactions = user_profile.loc[0, 'mean_interactions']
    base_pops = track_popularities.copy()
    base_pops = base_pops[base_pops['track_id'].isin(base)]
    
    base_num_recommendations = base_pops.shape[0]
    base_pop_counts = base_pops['popularity'].value_counts()
    base_head_ratio = base_pop_counts.get('head', 0) / base_num_recommendations
    base_mid_ratio = base_pop_counts.get('mid', 0) / base_num_recommendations
    base_tail_ratio = base_pop_counts.get('tail', 0) / base_num_recommendations
    
    base_mean_interactions = base_pops['interactions'].mean()
    base_median_interactions = base_pops['interactions'].median()
    
    base_popularity_lift = (base_mean_interactions - profile_mean_interactions) / profile_mean_interactions
    
    base_jensen_shannon = jensen_shannon(
        {'head_ratio': base_head_ratio, 'mid_ratio': base_mid_ratio, 'tail_ratio': base_tail_ratio},
        user_profile
    )
    
    # CP metrics
    cp_pops = track_popularities.copy()
    cp_pops = cp_pops[cp_pops['track_id'].isin(cp)]
    
    cp_num_recommendations = cp_pops.shape[0]
    cp_pop_counts = cp_pops['popularity'].value_counts()
    cp_head_ratio = cp_pop_counts.get('head', 0) / cp_num_recommendations
    cp_mid_ratio = cp_pop_counts.get('mid', 0) / cp_num_recommendations
    cp_tail_ratio = cp_pop_counts.get('tail', 0) / cp_num_recommendations
    
    cp_mean_interactions = cp_pops['interactions'].mean()
    cp_median_interactions = cp_pops['interactions'].median()
    
    cp_popularity_lift = (cp_mean_interactions - profile_mean_interactions) / profile_mean_interactions
    
    cp_jensen_shannon = jensen_shannon(
        {'head_ratio': cp_head_ratio, 'mid_ratio': cp_mid_ratio, 'tail_ratio': cp_tail_ratio},
        user_profile
    )
    
    # Fair metrics
    fair_pops = track_popularities.copy()
    fair_pops = fair_pops[fair_pops['track_id'].isin(fair)]
    
    fair_num_recommendations = fair_pops.shape[0]
    fair_pop_counts = fair_pops['popularity'].value_counts()
    fair_head_ratio = fair_pop_counts.get('head', 0) / fair_num_recommendations
    fair_mid_ratio = fair_pop_counts.get('mid', 0) / fair_num_recommendations
    fair_tail_ratio = fair_pop_counts.get('tail', 0) / fair_num_recommendations
    
    fair_mean_interactions = fair_pops['interactions'].mean()
    fair_median_interactions = fair_pops['interactions'].median()
    
    fair_popularity_lift = (fair_mean_interactions - profile_mean_interactions) / profile_mean_interactions
    
    fair_jensen_shannon = jensen_shannon(
        {'head_ratio': fair_head_ratio, 'mid_ratio': fair_mid_ratio, 'tail_ratio': fair_tail_ratio},
        user_profile
    )

    # Create recommendation_stats dataframe
    recommendation_stats = pd.DataFrame({
        'algorithm': ['base', 'cp', 'fair'],
        'user_id': [user_id, user_id, user_id],
        'mean_interactions': [base_mean_interactions, cp_mean_interactions, fair_mean_interactions],
        'median_interactions': [base_median_interactions, cp_median_interactions, fair_median_interactions],
        'popularity_lift': [base_popularity_lift, cp_popularity_lift, fair_popularity_lift],
        'jensen_shannon': [base_jensen_shannon, cp_jensen_shannon, fair_jensen_shannon],
        'head_ratio': [base_head_ratio, cp_head_ratio, fair_head_ratio],
        'mid_ratio': [base_mid_ratio, cp_mid_ratio, fair_mid_ratio],
        'tail_ratio': [base_tail_ratio, cp_tail_ratio, fair_tail_ratio]
    })
    # Convert the lists to sets
    base_set = set(base)
    cp_set = set(cp)
    fair_set = set(fair)
    
    # Calculate Jaccard similarity
    base_cp_similarity = len(base_set.intersection(cp_set)) / len(base_set.union(cp_set))
    base_fair_similarity = len(base_set.intersection(fair_set)) / len(base_set.union(fair_set))
    cp_fair_similarity = len(cp_set.intersection(fair_set)) / len(cp_set.union(fair_set))
    
    # Create similarity_stats dataframe
    similarity_stats = pd.DataFrame({
        'algorithm_1': ['base', 'base', 'cp'],
        'algorithm_2': ['cp', 'fair', 'fair'],
        'user_id': [user_id, user_id, user_id],
        'jaccard_similarity': [base_cp_similarity, base_fair_similarity, cp_fair_similarity]
    })
    
    
    
    return recommendation_stats, similarity_stats
    
    

def jensen_shannon(recommendation_ratios, user_profile):
    
    """
    Formula for computing Jensen-Shannon divergence
    """
    epsilon = 1e-8  # Small non-zero value
    A = 0
    B = 0

    for c in ['head_ratio', 'mid_ratio', 'tail_ratio']:
        profile_ratio = user_profile.loc[0, c]

        recommended_ratio = recommendation_ratios[c]

        if profile_ratio == 0:
            profile_ratio += epsilon

        if recommended_ratio == 0:
            recommended_ratio += epsilon

        A += profile_ratio * math.log2((2 * profile_ratio) / (profile_ratio + recommended_ratio))
        B += recommended_ratio * math.log2((2 * recommended_ratio) / (profile_ratio + recommended_ratio))

    js = (A + B) / 2
    return js



def analyse_interaction_data(user_id, condition, time_spent, track_uris, ranks):
    """
    Computing interaction stats about the time and choices of the user
    """
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    time_stats = pd.DataFrame({'user_id': [user_id],
                               'condition': [condition],
                               'interaction_time': [current_datetime_str],
                               'time_spent':[time_spent],
                               })
    track_choices = pd.DataFrame({'user_id':[user_id],
                                 'condition':[condition],
                                 'rank': [ranks],
                                 'uri': [track_uris],
                                 })


    return time_stats, track_choices