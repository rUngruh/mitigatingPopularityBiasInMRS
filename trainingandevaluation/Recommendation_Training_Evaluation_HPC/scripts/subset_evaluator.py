# -*- coding: utf-8 -*-
"""
Due to the large number of items, it was not possible to rerank each item. 
A subset of 250 users was chosen and the reranking was done for these users.

"""

import sys
import os

import numpy as np
from Preprocessing import data_preprocessing as pre

import pandas as pd
import pickle
import csv
from scipy.sparse import save_npz, load_npz


from model_handler import predict, load_model, rerank
from Evaluation.evaluation import user_popularity_deviation, popularity_lift, descriptive_popularity_measures, evaluate, \
    group_average_popularity, Gini, aggregate_diversity, average_cov_long_tail_items, average_perc_long_tail_items,\
        plot_popularity_distributions


# Get the input directory and output directory from command-line arguments
input_directory = sys.argv[1]
output_directory = sys.argv[2]
model_directory = sys.argv[3]
ev_num = sys.argv[4]
test_num = int(sys.argv[5])

# Check if the input directory exists
if not os.path.isdir(input_directory):
    print("Input directory does not exist.")
    
# Get the path of the input files
trainPath = os.path.join(input_directory, "A_train.npz")
testPath = os.path.join(input_directory, "A_test.npz")
mappingPath = os.path.join(input_directory, "init_mappings.npz")

# Load the train and test matrices
test, train, mappings = pre.load_train_and_test_matrix(dataPathTest = testPath, dataPathTrain = trainPath, dataPathMappings = mappingPath)
user_index_map, track_index_map = mappings


# Get the path of the files to be loaded that save the profile data
profilePath = input_directory + '/testset/'

# Set the paths of the files to be saved
metricPath = output_directory + '/Evaluation' + ev_num + '_' + str(test_num) + '.csv'

# Load the user and track mappings
trackDataPath = os.path.join(input_directory, "track_popularity.csv")
userDataPath = os.path.join(input_directory, "user_profiles.csv")

modelPath = model_directory
    

algorithmName = ''

# Load preprocessed information about the profiles
profile_popularity_ratios = pd.read_csv(profilePath + 'profile_popularity_ratios.csv')
profile_average_popularity = pd.read_csv(profilePath + 'profile_average_popularity.csv')

# Load profile_track_ids
with open(profilePath + 'profile_track_ids.pkl', 'rb') as f:
    profile_track_ids = pickle.load(f)

# Load profile_popularity_means
with open(profilePath + 'profile_popularity_means.pkl', 'rb') as f:
    profile_popularity_means = pickle.load(f)

# Load profile_popularity_medians
with open(profilePath + 'profile_popularity_medians.pkl', 'rb') as f:
    profile_popularity_medians = pickle.load(f)

# Load profile_popularity_variances
with open(profilePath + 'profile_popularity_variances.pkl', 'rb') as f:
    profile_popularity_variances = pickle.load(f)
    
# Load the track and user data
track_popularity = pd.read_csv(trackDataPath)
user_profiles = pd.read_csv(userDataPath)

import random
#redefine user_idxs to select a limited number of items and users
user_idxs = list(np.random.choice(test.shape[0], 250, replace=False))


#redefine test
test = test[user_idxs]

#redefine user_ids
user_ids = [user_index_map[uidx] for uidx in user_idxs]

# redefine profile_popularity_ratios and profile_average_popularity
profile_popularity_ratios = profile_popularity_ratios[profile_popularity_ratios['user_id'].isin(user_ids)]
profile_average_popularity = profile_average_popularity[profile_average_popularity['user_id'].isin(user_ids)]

# Compute the number of users and tracks
num_users = test.shape[0]
num_tracks = test.shape[1]


# Initialize the DataFrame
df = pd.DataFrame(columns=[
    "Algorithm", "precision", "recall", 
    "ndcg", "map", "mean", "median", "variance",
    "gap_blockbuster", "gap_diverse", "gap_niche", "aptl", "actl",
    "agg_div", "gini", "upd", "upd_blockbuster", "upd_diverse",
    "upd_niche", "pl", "pl_blockbuster", "pl_diverse", "pl_niche"
])

# Set initial values for all parameters
N=''
k=''
    
modelName = ''
delta = 0
p_fair = 0
alpha_fair = 0.1
#ranker=None
ranker = None
reranker = ''

recommended_track_ids_list, recommended_track_idxs_list, recommended_track_scores_list = [], [], []
recommended_popularity_ratios, recommended_average_popularity, recommended_popularity_means, recommended_popularity_medians, recommended_popularity_variances = pd.DataFrame(), pd.DataFrame(), [], [], []




def set_algorithmName():
    """
    retrieves the global algorithmName object and set the name depending on the model and the mitigation algorithm as well as the parameters
    """
    
    global algorithmName
    algorithmName = modelName + '@' + str(N)
    if reranker != '':
        if reranker != 'FAIR':
            algorithmName += '_' + reranker + '@' + str(k) + f'd{delta:.4f}'
        else:
            algorithmName += f'_{reranker}@{k}alpha{alpha_fair:.4f}p{p_fair:.4f}'
        

def save_metrics(metric_values):
    # Create a dictionary of metric values
    row_data = {
        "Algorithm": metric_values["Algorithm"],
        "precision": metric_values["precision"],
        "recall": metric_values["recall"],
        "ndcg": metric_values["ndcg"],
        "map": metric_values["map"],
        "mean": metric_values["mean"],
        "median": metric_values["median"],
        "variance": metric_values["variance"],
        "gap_blockbuster": metric_values["gap_blockbuster"],
        "gap_diverse": metric_values["gap_diverse"],
        "gap_niche": metric_values["gap_niche"],
        "aptl": metric_values["aptl"],
        "actl": metric_values["actl"],
        "agg_div": metric_values["agg_div"],
        "gini": metric_values["gini"],
        "upd": metric_values["upd"],
        "upd_blockbuster": metric_values["upd_blockbuster"],
        "upd_diverse": metric_values["upd_diverse"],
        "upd_niche": metric_values["upd_niche"],
        "pl": metric_values["pl"],
        "pl_blockbuster": metric_values["pl_blockbuster"],
        "pl_diverse": metric_values["pl_diverse"],
        "pl_niche": metric_values["pl_niche"]
    }

    # Add a row to the DataFrame
    df.loc[len(df)] = row_data

    # Save the DataFrame to the specified path
    df.to_csv(metricPath, index=False)

def create_recommendations(recommender, N=N):
    
    """
    Creates N recommendations for all users in the test set. 
    """
    
    # Set batch size and number of batches
    batch_size = 50
    num_batches = int(np.ceil(num_users / batch_size))
    
    print('Compute recommendations...')
    recommended_track_ids_list = []
    recommended_track_idxs_list = []
    recommended_track_scores_list = []
    i = 0

    # Iterate over all batches and create recommendations for each user in the batch
    for batch_num in range(num_batches):
        # Get the user indices for the current batch
        user_batch = user_idxs[batch_num*batch_size:(batch_num+1)*batch_size]
        
        # Call the predict function to create recommendations for the current batch
        recommendations_idxs, recommended_track_ids, scores = predict(recommender, user_idxs=user_batch, N=N, user_index_map=user_index_map, track_index_map=track_index_map)
        
        # Store recommended_track_ids for each user in the batch
        recommended_track_idxs_list.extend(recommendations_idxs)
        recommended_track_ids_list.extend(recommended_track_ids)
        recommended_track_scores_list.extend(scores)
        
        i += len(user_batch)
        print(f'Progress: {(i / num_users * 100)} %')
        
    return recommended_track_ids_list, recommended_track_idxs_list, recommended_track_scores_list

def compute_standard_recommender_metrics():
    """
    Computes standard popularity metrics for the recommended tracks for each user that will be used multiple times during the evaluation.
    """
    # Initialize lists that will be used to store the statistics
    recommended_popularity_means = []
    recommended_popularity_medians = []
    recommended_popularity_variances = []

    recommended_head_ratio = []
    recommended_mid_ratio = []
    recommended_tail_ratio = []
    
    print('Computing recommended distributions for each user...')
    i=0
    for recommended_track_ids_u in recommended_track_ids_list:
        if i % 500 == 0:
            print(f'Progress: {(i /num_users * 100)} %')
        # Calculate the popularity of the recommended tracks
        recommended_tracks = track_popularity[track_popularity['track_id'].isin(recommended_track_ids_u)]
        recommended_popularity = recommended_tracks['interactions']

        recommended_popularity_means.append(recommended_popularity.mean())
        recommended_popularity_medians.append(recommended_popularity.median())
        recommended_popularity_variances.append(recommended_popularity.var())

        track_counts = recommended_tracks['popularity'].value_counts()
        total_tracks = track_counts.sum()

        recommended_head_ratio.append(track_counts.get('head', 0) / total_tracks)
        recommended_mid_ratio.append(track_counts.get('mid', 0) / total_tracks)
        recommended_tail_ratio.append(track_counts.get('tail', 0) / total_tracks)

        i += 1

    
    # Store the statistics in DataFrames
    recommended_popularity_ratios = pd.DataFrame({
        'user_id': user_ids,
        'user_type': ['Overall'] * num_users,
        'head_ratio': recommended_head_ratio,
        'mid_ratio': recommended_mid_ratio,
        'tail_ratio': recommended_tail_ratio
    })
    
    
    
    # Store the statistics in DataFrames
    recommended_average_popularity = pd.DataFrame({
        'user_id': user_ids,
        'mean_popularity': recommended_popularity_means
    })
    
    return recommended_popularity_ratios, recommended_average_popularity, recommended_popularity_means, recommended_popularity_medians, recommended_popularity_variances


def compute_profile_metrics():
    
    """
    Computes popularity metrics of the user profiles for that will be used multiple times during the evaluation.
    """
    
    # Compute statistics
    print("Calculating descriptive popularity measures for profiles...")
    popularity_mean, popularity_median, popularity_variance = descriptive_popularity_measures(sum(profile_popularity_means), sum(profile_popularity_medians),
                                    sum(profile_popularity_variances), num_users)
    
    print("Calculating group average popularity for profiles...")
    blockbuster_avg_pop, diverse_avg_pop, niche_avg_pop= group_average_popularity(profile_average_popularity, user_profiles)

    print("Calculating average percentage of long-tail items for profiles...")
    aptl = average_perc_long_tail_items(profile_popularity_ratios, num_users)


    print("Calculating coverage of long tail items for profiles...")
    actl= average_cov_long_tail_items(profile_track_ids, track_popularity, num_tracks)
    
    print("Calculating aggregate diversity for profiles...")
    agg_div = aggregate_diversity(profile_track_ids, num_tracks)

    print("Calculating Gini for profiles...")
    gini = Gini(profile_track_ids, num_tracks)
    
    print("Plotting popularity distributions for user profiles...")
    plot_popularity_distributions(profile_popularity_ratios, output_directory + '/' + algorithmName, user_profiles)
    
    # Save in dataframe
    metric_values = {
    "Algorithm": algorithmName,
    "precision": np.nan,
    "recall": np.nan,
    "ndcg": np.nan,
    "map": np.nan,
    "mean": popularity_mean,
    "median": popularity_median,
    "variance": popularity_variance,
    "gap_blockbuster": blockbuster_avg_pop,
    "gap_diverse": diverse_avg_pop,
    "gap_niche": niche_avg_pop,
    "aptl": aptl,
    "actl": actl,
    "agg_div": agg_div,
    "gini": gini,
    "upd": np.nan,
    "upd_blockbuster": np.nan,
    "upd_diverse": np.nan,
    "upd_niche": np.nan,
    "pl": np.nan,
    "pl_blockbuster": np.nan,
    "pl_diverse": np.nan,
    "pl_niche": np.nan
    }
    
    save_metrics(metric_values)


def compute_recommender_metrics():    
    """
    Computes the performance metrics and other popularity metrics that rely on the standard metrics 
    for the recommendations and saves them in a dataframe.
    """
    
    # Compute performance metrics
    print('Performance metrics:')
    precision, recall, ndcg, map_score,  _, _, _, _ = evaluate(recommended_track_idxs_list, test, N if reranker=='' else k)
    
    print("Calculating descriptive popularity measures for recommendations...")
    popularity_mean, popularity_median, popularity_variance = descriptive_popularity_measures(sum(recommended_popularity_means), sum(recommended_popularity_medians),
                                    sum(recommended_popularity_variances), num_users)
    
    print("Calculating group average popularity for recommendations...")
    blockbuster_avg_pop, diverse_avg_pop, niche_avg_pop = group_average_popularity(recommended_average_popularity, user_profiles)

    print("Calculating average percentage of long-tail items for recommendations...")
    aptl = average_perc_long_tail_items(recommended_popularity_ratios, num_users)

    print("Calculating coverage of long tail items for recommendations...")
    actl= average_cov_long_tail_items(recommended_track_ids_list, track_popularity, num_tracks)
    
    print("Calculating aggregate diversity for recommendations...")
    agg_div =aggregate_diversity(recommended_track_ids_list, num_tracks)
    
    print("Calculating Gini for recommendations...")
    gini = Gini(recommended_track_ids_list, num_tracks)

    print("Calculating user popularity deviation...")
    upd, group_upd_list, _ = user_popularity_deviation(recommended_popularity_ratios, profile_popularity_ratios, user_profiles)
    upd_block, upd_div, upd_niche = group_upd_list
    
    print("Calculating popularity lift...")
    pl_all, pl_block, pl_div, pl_niche = popularity_lift(recommended_average_popularity, profile_average_popularity, user_profiles)
    
    print("Plotting popularity distributions for recommendations...")
    plot_popularity_distributions(recommended_popularity_ratios,   output_directory + '/' + algorithmName, user_profiles)
    
    # Save in dataframe
    metric_values = {
    "Algorithm": algorithmName,
    "precision": precision,
    "recall": recall,
    "ndcg": ndcg,
    "map": map_score,
    "mean": popularity_mean,
    "median": popularity_median,
    "variance": popularity_variance,
    "gap_blockbuster": blockbuster_avg_pop,
    "gap_diverse": diverse_avg_pop,
    "gap_niche": niche_avg_pop,
    "aptl": aptl,
    "actl": actl,
    "agg_div": agg_div,
    "gini": gini,
    "upd": upd,
    "upd_blockbuster": upd_block,
    "upd_diverse": upd_div,
    "upd_niche": upd_niche,
    "pl": pl_all,
    "pl_blockbuster": pl_block,
    "pl_diverse": pl_div,
    "pl_niche": pl_niche
    }
    
    save_metrics(metric_values)

def evaluation_step(modelName_, N_=10, rerankerName_='', k_=10, start_delta=0.0, end_delta=1.0):
    """
    Accomplishes one evaluation step by iteratively computing recommendations and evaluating them for a certain model and mitigation algorithm.
    """
    
    # Set the gobal variables, so that they can be accessed easily from the functions
    
    global N
    global k
    global delta
    global alpha_fair
    global p_fair
    global modelName
    global reranker
    global ranker

    
    global recommended_track_ids_list
    global recommended_track_idxs_list
    global recommended_track_scores_list
    global recommended_popularity_ratios
    global recommended_average_popularity
    global recommended_popularity_means
    global recommended_popularity_medians
    global recommended_popularity_variances
    global profile_popularity_ratios
    global average_popularity_ratios
    global profile_average_popularity
    global user_ids
    global user_idxs
    global test
    N=N_
    k=k_

    reranker = rerankerName_
        
    modelName = modelName_
    
    # Set the name of the algorithm
    set_algorithmName()
    
    print('Now processing: ' + algorithmName)

    # Since, profile metrics remain stable and will be used multiple times, they can be computed once by calling the model 'profiles'
    if modelName == 'profiles':
        compute_profile_metrics()
        
    # For all other models, the recommendations have to be computed and evaluated
    else:
        # Load the model
        ranker = load_model(modelName, modelPath=modelPath)
        
        # If a reranker/mitigation algorithm should be applied
        if reranker != '':
            
            # Since the reranking is applied on the same recommendations each time
            # they can be computed once and the corresponding metrics can be calculated
            # and saved to save time
            if reranker == 'init':
                init_recommended_track_ids_list, init_recommended_track_idxs_list, init_recommended_track_scores_list = create_recommendations(ranker, N=N)
                save_as_2D_csv(init_recommended_track_ids_list, os.path.join(model_directory, 'init_recommended_track_ids_list.csv'))
                save_as_2D_csv(init_recommended_track_idxs_list, os.path.join(model_directory, 'init_recommended_track_idxs_list.csv'))
                save_as_2D_csv(init_recommended_track_scores_list, os.path.join(model_directory, 'init_recommended_track_scores_list.csv'))
                save_as_1D_csv(user_ids, os.path.join(model_directory, 'user_ids.csv'))
                save_as_1D_csv(user_idxs, os.path.join(model_directory, 'user_idxs.csv'))
                save_npz(os.path.join(model_directory, 'test.npz'), test.tocsr())
                profile_popularity_ratios.to_csv(os.path.join(model_directory, 'profile_popularity_ratios.csv'), index=False)
                profile_average_popularity.to_csv(os.path.join(model_directory, 'profile_average_popularity.csv'), index=False)
                return
            
            # After the first time, we can just retrieve the initial recommendations and scores from the csv files
            profile_popularity_ratios = pd.read_csv(os.path.join(model_directory,'profile_popularity_ratios.csv'))
            profile_average_popularity = pd.read_csv(os.path.join(model_directory,'profile_average_popularity.csv'))
            user_ids = load_from_1D_csv(os.path.join(model_directory, 'user_ids.csv'))
            user_idxs = load_from_1D_csv(os.path.join(model_directory, 'user_idxs.csv'))
            test = load_npz(os.path.join(model_directory, 'test.npz'))
            init_recommended_track_ids_list = load_from_2D_csv(os.path.join(model_directory, 'init_recommended_track_ids_list.csv'))
            init_recommended_track_idxs_list = load_from_2D_csv(os.path.join(model_directory, 'init_recommended_track_idxs_list.csv'))
            init_recommended_track_scores_list = load_from_2D_csv(os.path.join(model_directory, 'init_recommended_track_scores_list.csv'))

            
            delta = start_delta
            
            # iterate various deltas
            while delta <= end_delta:
                
                # fair requires a minimum 0 of .02 and a maximum of .98
                if reranker =='FAIR':
                    p_fair = delta

                    if p_fair < 0.02:
                        p_fair = 0.02
                    if p_fair > 0.98:
                        p_fair = 0.98
                        
                # Compute algorithm name with the newly updated parameters
                set_algorithmName()
                
                # Rerank the initial recommendations
                recommended_track_ids_list, recommended_track_idxs_list, recommended_track_scores_list = \
                    rerank(reranker, init_recommended_track_idxs_list, init_recommended_track_ids_list, 
                           init_recommended_track_scores_list, k=k, user_ids=user_ids, 
                           user_profiles=user_profiles, track_popularities=track_popularity, delta=delta, p_fair=p_fair, alpha_fair=alpha_fair)
                
                #Compute the metrics for the reranked recommendations
                recommended_popularity_ratios, recommended_average_popularity, recommended_popularity_means, recommended_popularity_medians, recommended_popularity_variances = \
                    compute_standard_recommender_metrics()
                
                # Compute further recommendations
                compute_recommender_metrics()
                
                delta += 0.1

        # If no reranker/mitigation algorithm should be applied just compute the recommendations and evaluate them
        else:
            recommended_track_ids_list, recommended_track_idxs_list, recommended_track_scores_list = create_recommendations(ranker, N=N)
            recommended_popularity_ratios, recommended_average_popularity, recommended_popularity_means, recommended_popularity_medians, recommended_popularity_variances = \
                compute_standard_recommender_metrics()
            
            compute_recommender_metrics()
    


def save_as_2D_csv(data, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)     
        
def save_as_1D_csv(data, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)      
        
def load_from_2D_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = [[int(item) if item.isdigit() else float(item) for item in row] for row in reader]
    return data

def load_from_1D_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = [int(item) if item.isdigit() else float(item) for item in next(reader)]
    return data



# Evaluate the model and save the stats, each number is connected to one certain model and mitigation algorithm
if test_num == 1:
    evaluation_step(modelName_='profiles')
if test_num == 2:
    evaluation_step(modelName_='Random', N_=25)
if test_num == 3:
    evaluation_step(modelName_='Popularity', N_=25)   
if test_num == 4:
    evaluation_step(modelName_='RankALS', N_=100)
if test_num == 5:
    evaluation_step(modelName_='RankALS', N_=250, rerankerName_='XQ', k_=25, start_delta=0, end_delta=0.5)
if test_num == 6:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='XQ', k_=25, start_delta=0.6, end_delta=1.0)#almost 2 min per rerank
if test_num == 7:
    evaluation_step(modelName_='RankALS', N_=250, rerankerName_='FAIR', k_=25, start_delta=0, end_delta=0.5)
if test_num == 8:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='FAIR', k_=25, start_delta=0.6, end_delta=1.0)#16 sec per rerank
if test_num == 9:
    evaluation_step(modelName_='RankALS', N_=250, rerankerName_='CP', k_=25, start_delta=0, end_delta=0.5)
if test_num == 10:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='CP', k_=25, start_delta=0.6, end_delta=1.0)#1:40 per rerank


if test_num == 101:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='XQ', k_=25, start_delta=0, end_delta=0.1) # will take approx 400 min -> 8 hrs
if test_num == 102:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='XQ', k_=25, start_delta=0.2, end_delta=0.3)
if test_num == 103:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='XQ', k_=25, start_delta=0.4, end_delta=0.5)
if test_num == 104:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='XQ', k_=25, start_delta=0.6, end_delta=0.7)
if test_num == 105:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='XQ', k_=25, start_delta=0.8, end_delta=0.9)
if test_num == 106:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='XQ', k_=25, start_delta=1.0, end_delta=1.0)
if test_num == 107:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='XQ', k_=25, start_delta=0.3, end_delta=0.3)
if test_num == 111:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='FAIR', k_=25, start_delta=0, end_delta=0.2) # will take approx 4 hrs
if test_num == 112:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='FAIR', k_=25, start_delta=0.3, end_delta=0.5)
if test_num == 113:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='FAIR', k_=25, start_delta=0.6, end_delta=0.8)
if test_num == 114:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='FAIR', k_=25, start_delta=0.9, end_delta=1.0)
if test_num == 121:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='CP', k_=25, start_delta=0, end_delta=0.1) # will take approx 400 min -> 8 hrs
if test_num == 122:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='CP', k_=25, start_delta=0.2, end_delta=0.3)
if test_num == 123:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='CP', k_=25, start_delta=0.4, end_delta=0.5)
if test_num == 124:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='CP', k_=25, start_delta=0.6, end_delta=0.7)
if test_num == 125:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='CP', k_=25, start_delta=0.8, end_delta=0.9)
if test_num == 126:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='CP', k_=25, start_delta=1.0, end_delta=1.0)
if test_num == 127:
    evaluation_step(modelName_='RankALS', N_=5000, rerankerName_='CP', k_=25, start_delta=0.3, end_delta=0.3)



if test_num == 11:
    evaluation_step(modelName_='RankALS1', N_=25)
if test_num == 12:
    evaluation_step(modelName_='RankALS2', N_=25)
if test_num == 13:
    evaluation_step(modelName_='RankALS3', N_=25)
if test_num == 14:
    evaluation_step(modelName_='RankALS4', N_=25)
if test_num == 15:
    evaluation_step(modelName_='RankALS5', N_=25)
if test_num == 16:
    evaluation_step(modelName_='RankALS6', N_=25)
if test_num == 17:
    evaluation_step(modelName_='RankALS7', N_=25)
if test_num == 18:
    evaluation_step(modelName_='RankALS8', N_=25)
if test_num == 19:
    evaluation_step(modelName_='RankALS9', N_=25)
if test_num == 20:
    evaluation_step(modelName_='RankALS10', N_=25)
if test_num == 21:
    evaluation_step(modelName_='RankALS11', N_=25)
if test_num == 22:
    evaluation_step(modelName_='RankALS12', N_=25)    
if test_num == 23:
    evaluation_step(modelName_='RankALS13', N_=25)
if test_num == 24:
    evaluation_step(modelName_='RankALS14', N_=25)
if test_num == 25:
    evaluation_step(modelName_='RankALS15', N_=25)
if test_num == 26:
    evaluation_step(modelName_='RankALS16', N_=25)
if test_num == 27:
    evaluation_step(modelName_='RankALS17', N_=25)
if test_num == 28:
    evaluation_step(modelName_='RankALS18', N_=25)
if test_num == 29:
    evaluation_step(modelName_='RankALS19', N_=25)
if test_num == 30:
    evaluation_step(modelName_='RankALS20', N_=25)
if test_num == 31:
    evaluation_step(modelName_='RankALS21', N_=25)
if test_num == 51:
    evaluation_step(modelName_='RankALS128_5', N_=25)
if test_num == 52:
    evaluation_step(modelName_='RankALS128_10', N_=25)
if test_num == 53:
    evaluation_step(modelName_='RankALS128_20', N_=25)


print('Finished Evaluation')
