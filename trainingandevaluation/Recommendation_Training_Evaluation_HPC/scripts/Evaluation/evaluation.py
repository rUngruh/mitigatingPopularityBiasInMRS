# -*- coding: utf-8 -*-



import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


def save_metric_values(metric_name, algorithm_name, metric_values):
    """
    Saves the metric values for the algorithm in a CSV file.
    """
    
    
    # Create a filename based on the metric name
    filename = metric_name + ".csv"

    # Check if the file already elet'xists
    if os.path.exists(filename):
        # If the file exists, load it into a DataFrame
        df = pd.read_csv(filename, index_col=0)
    else:
        # If the file doesn't exist, create a new DataFrame
        df = pd.DataFrame()

    # Add a new column with the metric values for the algorithm
    df[algorithm_name] = metric_values

    # Save the DataFrame to the CSV file
    df.to_csv(filename)
    
    
def preprocess_profile_data(train_user_item_matrix, user_index_map, track_index_map, trackDataPath, userDataPath, savePath):
    """
    Preprocesses the user profile data and saves it to the specified path.
    """
    
    # Get the track_popularities
    track_popularity = pd.read_csv(trackDataPath)

    # Get the user_profiles
    profile_indices = []
    for row in train_user_item_matrix:
        profile_indices.append(row.nonzero()[1])

    # Get the track_ids for each user
    print('get track ids...')
    profile_track_ids = [[track_index_map[index] for index in indices] for indices in profile_indices]
    valid_user_idx = []
    profile_popularity_means = []
    profile_popularity_medians = []
    profile_popularity_variances = []
    print('compute means, medians, variances...')
    
    # For each user, compute the mean, median and variance of the popularity of the tracks in their profile
    for user_idx in range(train_user_item_matrix.shape[0]):
        if user_idx % 1000 == 0:
            print(f'Progress: {(user_idx / train_user_item_matrix.shape[0] * 100):.4f} %')
        user_track_ids = profile_track_ids[user_idx]
        
        if len(user_track_ids) < 2:
            continue
        valid_user_idx.append(user_idx)
        user_tracks = track_popularity[track_popularity['track_id'].isin(user_track_ids)]
        user_track_popularity = user_tracks['interactions']
        
        profile_popularity_means.append(user_track_popularity.mean())
        profile_popularity_medians.append(user_track_popularity.median())
        profile_popularity_variances.append(user_track_popularity.var())
    
    # Create a DataFrame with the user_id and the mean, median and variance of the popularity of the tracks in their profile
    print('create frame for user profile ratios')
    profile_popularity_ratios = pd.DataFrame({
        'user_id': [user_index_map[idx] for idx in valid_user_idx],
        'user_type' :['Overall'] * len(valid_user_idx),
        'head_ratio': np.zeros(len(valid_user_idx)),
        'mid_ratio': np.zeros(len(valid_user_idx)),
        'tail_ratio': np.zeros(len(valid_user_idx))
    })
    
    
    profile_popularity_ratios['head_ratio'] = np.nan
    profile_popularity_ratios['mid_ratio'] = np.nan
    profile_popularity_ratios['tail_ratio'] = np.nan
    
    track_popularity_labels = track_popularity['popularity'].unique()
    
    # For each user, compute the  head, mid, and tail ratio of the items in their user profile
    print('compute ratios for each user...')
    for i, idx in enumerate(valid_user_idx):
        user_track_ids = profile_track_ids[idx]
        if i % 1000 == 0:
            print(f'Progress: {(i / len(profile_track_ids) * 100):.4f} %')
        user_tracks = track_popularity[track_popularity['track_id'].isin(user_track_ids)]
        track_counts = user_tracks['popularity'].value_counts()
        total_tracks = track_counts.sum()
        
        for label in track_popularity_labels:
            ratio = track_counts.get(label, 0) / total_tracks
            profile_popularity_ratios.loc[i, label + '_ratio'] = ratio
    
    # Create a DataFrame with the user_id and the mean popularity of the tracks in their profile
    profile_average_popularity = pd.DataFrame({
        'user_id': [user_index_map[idx] for idx in valid_user_idx],
        'mean_popularity': profile_popularity_means
    })
    
    # Save preprocessed data
    print('save data...')
    # Save preprocessed data
    profile_popularity_ratios.to_csv(savePath + 'profile_popularity_ratios.csv', index=False)
    profile_average_popularity.to_csv(savePath + 'profile_average_popularity.csv', index=False)
    
    with open(savePath + 'profile_track_ids.pkl', 'wb') as f:
        pickle.dump(profile_track_ids, f)

    with open(savePath + 'profile_popularity_means.pkl', 'wb') as f:
        pickle.dump(profile_popularity_means, f)
        
    with open(savePath + 'profile_popularity_medians.pkl', 'wb') as f:
        pickle.dump(profile_popularity_medians, f)
    
    with open(savePath + 'profile_popularity_variances.pkl', 'wb') as f:
        pickle.dump(profile_popularity_variances, f)

    
    
def calc_precision(recommended_indices, true_indices, n):
    """
    Computes the precision for the given recommendations and true indices.
    """
    relevant_recommended = recommended_indices[:n]
    num_common = len(set(relevant_recommended) & set(true_indices))
    precision = num_common / n

    
    return precision


def calc_recall(recommended_indices, true_indices, n):
    """
    Computes the recall for the given recommendations and true indices.
    """
    relevant_recommended = recommended_indices[:n]
    num_common = len(set(relevant_recommended) & set(true_indices))
    recall = num_common / len(true_indices)

    return recall


def calc_ndcg(recommended_indices, true_indices, n):
    #formula based on: http://ethen8181.github.io/machine-learning/recsys/2_implicit.html#NDCG
    """
    Computes the NDCG for the given recommendations and true indices.
    """
    relevant_recommended = recommended_indices[:n]
    
    # Calculate the Discounted Cumulative Gain (DCG)
    
    dcg = 0
    for i, idx in enumerate(relevant_recommended):
        relevance = 1 if idx in true_indices else 0
        dcg += (2 ** relevance - 1) / np.log2(i + 2)
    
    # Calculate the Ideal DCG (IDCG) as if all true items were recommended
    idcg = 0
    for i in range(min(n, len(true_indices))):
        idcg += (2 ** 1 - 1) / np.log2(i + 2)
    
    # Calculate the Normalized DCG (NDCG)
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return ndcg

def calc_average_precision(recommended_indices, true_indices, n):
    """
    Computes the average precision for the given recommendations and true indices.
    """
    relevant_recommended = recommended_indices[:n]
    num_common = len(set(relevant_recommended) & set(true_indices))

    precision_sum = 0
    relevant_count = 0
    for i, idx in enumerate(relevant_recommended):
        if idx in true_indices:
            relevant_count += 1
            precision = relevant_count / (i + 1)
            precision_sum += precision

    average_precision = precision_sum / min(n, len(true_indices))
    return average_precision


def evaluate(recommendations_indices, test_user_item_matrix, n, log=False, algorithmName=''):
    """
    Computes various performance metrics by calling the respective functions.
    """
    num_users = test_user_item_matrix.shape[0]
    precision_list = []
    recall_list = []
    ndcg_list = []
    map_list = []



    for user_idx in range(num_users):
        
        true_indices = test_user_item_matrix[user_idx].nonzero()[1]
        rec_idxs = recommendations_indices[user_idx]
        
        precision = calc_precision(rec_idxs, true_indices, n)
        recall = calc_recall(rec_idxs, true_indices, n)

        precision_list.append(precision)
        recall_list.append(recall)

        ndcg_list.append(calc_ndcg(rec_idxs, true_indices, n))
        map_list.append(calc_average_precision(rec_idxs, true_indices, n))

        
    ndcg = sum(ndcg_list) / num_users
    precision = sum(precision_list) / num_users
    recall = sum(recall_list) / num_users
    map_score = sum(map_list) / num_users

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'NDCG: {ndcg:.4f}')
    print(f'Mean Average Precision (MAP): {map_score:.4f}')

    
    return precision, recall, ndcg, map_score, precision_list, recall_list, ndcg_list, map_list


import math

def user_popularity_deviation(recommended_popularity_ratios, profile_popularity_ratios, user_profiles, log=False, algorithmName=''):
    """
    Computes the user popularity deviation for the given recommendations and user profile
    """
    group_upd_list = [] 
    group_js_list = []
    for u_type in ['Blockbuster-focused', 'Diverse', 'Niche']:
        gupd, g_js_list = group_user_popularity_deviation(recommended_popularity_ratios, profile_popularity_ratios, user_profiles, u_type, algorithmName=algorithmName)
        group_upd_list.append(gupd)
        group_js_list.append(g_js_list)
    upd = sum(group_upd_list) / 3
    print(f'User popularity deviation: {upd:.4f}')
    
    return upd, group_upd_list, group_js_list
    
def group_user_popularity_deviation(recommended_popularity_ratios, profile_popularity_ratios, user_profiles, user_type, log=False, algorithmName=''):
    """
    Computes the user popularity deviation for the given recommendations and user profile based on the jensen-shannon divergence
    
    """
    up = user_profiles.drop(['head_ratio', 'mid_ratio', 'tail_ratio'], axis=1)
    
    user_ids = up[(up['user_type'] == user_type) & (up['user_id'].isin(profile_popularity_ratios['user_id']))]['user_id']

    group_js_list = []
    for uid in user_ids:
        group_js_list.append(jensen_shannon(recommended_popularity_ratios, profile_popularity_ratios, uid))
    upd_g = sum(group_js_list) / len(user_ids)
    
    print(f'User group deviation for {user_type}  users:  {upd_g:.4f}')
    return upd_g, group_js_list
    
def jensen_shannon(recommended_popularity_ratios, profile_popularity_ratios, user_id):
    """
    Computes the Jensen-Shannon divergence for the given recommendations and user profile
    """
    epsilon = 1e-8  # Small non-zero value
    
    A = 0
    B = 0
    for c in ['head_ratio', 'mid_ratio', 'tail_ratio']:
        
        profile_ratio = profile_popularity_ratios[profile_popularity_ratios['user_id']==user_id][c].iloc[0]
        recommended_ratio = recommended_popularity_ratios[recommended_popularity_ratios['user_id']==user_id][c].iloc[0]
        
        if profile_ratio == 0:
            profile_ratio += epsilon
        
        if recommended_ratio == 0:
            recommended_ratio += epsilon
    
        A += profile_ratio * math.log2((2 * profile_ratio) / (profile_ratio + recommended_ratio))
        B += recommended_ratio * math.log2((2 * recommended_ratio) / (profile_ratio + recommended_ratio))
        
    js = (A + B) / 2
    return js

def average_cov_long_tail_items(recommended_track_ids, track_popularity, num_tracks, log=False, algorithmName=''):
    """
    Computes the average coverage of long tail items for the given recommendations
    """
    all_track_ids = [tid for user_tracks in recommended_track_ids for tid in user_tracks]
    
    unique_tracks = set(all_track_ids)

    num_tail_items = len(track_popularity[(track_popularity['track_id'].isin(unique_tracks)) & (track_popularity['popularity'] == 'tail')])
    
    actl = num_tail_items / len(track_popularity[track_popularity['popularity']=='tail'])
    
    print(f'Average Coverage of Long Tail Items: {actl:.4f}')
    return actl

def popularity_lift(recommended_average_popularity, profile_average_popularity, user_profiles, log=False, algorithmName=''):
    """
    Computes the popularity lift for the given recommendations and user profile
    """
    gap_r_block, gap_r_div, gap_r_niche = group_average_popularity(recommended_average_popularity, user_profiles, algorithmName=algorithmName)
    gap_p_block, gap_p_div, gap_p_niche = group_average_popularity(profile_average_popularity, user_profiles, algorithmName='User profile')

    arp_r = recommended_average_popularity['mean_popularity'].mean()
    arp_p = profile_average_popularity['mean_popularity'].mean()

    pl_all = (arp_r - arp_p) / arp_p
    pl_block = (gap_r_block - gap_p_block) / gap_p_block
    pl_div = (gap_r_div - gap_p_div) / gap_p_div
    pl_niche = (gap_r_niche - gap_p_niche) / gap_p_niche
    print('Popularity lift per group:')
    print(f'Overall: {pl_all:.4f}')
    print(f'Blockbuster-focused: {pl_block:.4f}')
    print(f'Diverse: {pl_div:.4f}')
    print(f'Niche: {pl_niche:.4f}')
            
    return pl_all, pl_block, pl_div, pl_niche

def average_perc_long_tail_items(popularity_ratios, num_users, log=False, algorithmName=''):
    """
    Computes the average percentage of long tail items for the given recommendations
    """
    aptl = popularity_ratios['tail_ratio'].mean()
    
    print(f'Average Percentage of Long Tail Items: {aptl:.4f}')
    return aptl
            
from collections import Counter
   
def Gini(track_ids, num_items, log=False, algorithmName=''):
    """
    Computes the Gini-index for the given recommendations
    """
    print('calculate Gini')
    #print(track_ids)
    flattened_track_ids = [tid for user_tracks in track_ids for tid in user_tracks]
    sum_ratio = 0
    counts = list(Counter(flattened_track_ids).values())
    counts += [0] * (num_items - len(counts))
    L_len = sum(counts)
    
    counts.sort()
    occ_sum = 0
    for k, count in enumerate(counts):
        
        occ =  count / L_len
        occ_sum += occ
        sum_ratio += ((num_items - (k+1) + 1) / num_items) * occ
    
    gini = 1 - ((2 / occ_sum) * sum_ratio)
    
    print(f'Gini-index: {gini:.4f}')
    return gini

def aggregate_diversity(recommended_track_ids, num_tracks, log=False, algorithmName=''):
    """
    Computes the aggregate diversity for the given recommendations
    """
    all_track_ids = [tid for user_tracks in recommended_track_ids for tid in user_tracks]
    
    num_unique_tracks = len(set(all_track_ids))
    
    agg_div = num_unique_tracks / num_tracks
    
    print(f'Aggregate Diversity: {agg_div:.4f}')
    return agg_div
    
def group_average_popularity(average_popularity, user_profiles, log=False, algorithmName=''):
    """
    Computes the average popularity for the given recommendations and user profile
    """
    up = user_profiles.drop(['head_ratio', 'mid_ratio', 'tail_ratio'], axis=1)
    ap = pd.merge(average_popularity, up, on='user_id')
    
    blockbuster_avg_pop = ap[ap['user_type'] == 'Blockbuster-focused']['mean_popularity'].mean()
    
    diverse_avg_pop = ap[ap['user_type'] == 'Diverse']['mean_popularity'].mean()
    niche_pop = ap[ap['user_type'] == 'Niche']['mean_popularity'].mean()
    
    print('Group Average Popularity:')
    print(f'Blockbuster-focused: {blockbuster_avg_pop:.4f}')
    print(f'Diverse: {diverse_avg_pop:.4f}')
    print(f'Niche: {niche_pop:.4f}')
    
    return (blockbuster_avg_pop, diverse_avg_pop, niche_pop)
    
def descriptive_popularity_measures(popularity_mean_sum, popularity_median_sum, popularity_variance_sum, num_users, log=False, algorithmName=''):
    """
    Computes the mean, median and variance of the popularity of the tracks in the user profiles
    """
    popularity_mean = popularity_mean_sum / num_users
    popularity_median = popularity_median_sum / num_users
    popularity_variance = popularity_variance_sum / num_users
    
    print(f"Mean Popularity (Average Recommendation Popularity): {popularity_mean:.4f}")
    print(f"Median Popularity: {popularity_median:.4f}")
    print(f"Variance Popularity: {popularity_variance:.4f}")
    
    return popularity_mean, popularity_median, popularity_variance

def plot_popularity_distributions(popularity_ratios, savePath, user_profiles, log=False, algorithmName=''):
     """
     Plots the popularity distributions for the given recommendations and user profile
     """
     # Set up the figure and axis
     fig, ax = plt.subplots(figsize=(3, 6))
     
     # Accumulate the values for each category
     popularity_ratios['mid_head'] = popularity_ratios['head_ratio'] + popularity_ratios['mid_ratio']
     popularity_ratios['tail_mid_head'] = popularity_ratios['mid_head'] + popularity_ratios['tail_ratio']
     
     # Create the stacked barplot with the specified colors
     cm = {'r': 'red', 'g': 'green', 'b': 'blue'}  # Define color map
     
     sns.barplot(y='tail_mid_head', data=popularity_ratios, ax=ax, label='Tail', color=cm['r'], errorbar=None)
     sns.barplot(y='mid_head', data=popularity_ratios, ax=ax, label='Mid', color=cm['g'], errorbar=None)
     sns.barplot(y='head_ratio', data=popularity_ratios, ax=ax, label='Head', color=cm['b'], errorbar=None)
     
     
     ax.set_ylabel("Ratio")
     ax.legend(title=algorithmName)
     sns.move_legend(
         ax, "lower center",
         bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
     )
     
     # Save the plot as an image file
     plot_path = savePath
     plt.savefig(plot_path + algorithmName + '_overall_distribution.png')  # Provide the desired file name and extension
     
     pr = popularity_ratios.drop(['user_type'], axis=1)
     up = user_profiles.drop(['head_ratio', 'mid_ratio', 'tail_ratio'], axis=1)
     
     # Merge user_profiles with user_types based on their index
     pr = pd.merge(pr, up, on='user_id')
     
     # Set up the figure and axis
     fig, ax = plt.subplots(figsize=(6, 6))
     
     x_axis_order = ['Blockbuster-focused', 'Diverse', 'Niche']
     
     # Create the stacked barplot with the specified colors
     cm = {'r': 'red', 'g': 'green', 'b': 'blue'}  # Define color map
     sns.barplot(x='user_type', y='tail_mid_head', data=pr, ax=ax, label='Tail', color=cm['r'], errorbar=None, order=x_axis_order)
     sns.barplot(x='user_type', y='mid_head', data=pr, ax=ax, label='Mid', color=cm['g'], errorbar=None, order=x_axis_order)
     sns.barplot(x='user_type', y='head_ratio', data=pr, ax=ax, label='Head', color=cm['b'], errorbar=None, order=x_axis_order)
     
     ax.set_xlabel("User Type")
     ax.set_ylabel("Ratio")
     ax.legend(title=algorithmName)
     sns.move_legend(
         ax, "lower center",
         bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
     )
     
     # Save the plot as an image file
     plot_path = savePath
     plt.savefig(plot_path + algorithmName + '_distribution_by_user_type.png')  # Provide the desired file name and extension


     print('plotted distributions...')


    