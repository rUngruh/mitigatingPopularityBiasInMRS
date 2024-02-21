# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from datetime import datetime
import random
import requests


from LFMRecommendations.TrackProcessing.URIprocessing import spids_to_ids, ids_to_spids, uris_to_spids
from LFMRecommendations.TrackProcessing.LFMFiltering import get_valid_spids
from LFMRecommendations.Recommending.Reranker import rerank
from LFMRecommendations.UserProcessing.profileCreation import create_user_profile

from LFMRecommendations.TrackProcessing.TrackAnalysis import analyse_recommendations
from Application.logging import write_recommendations_stats, write_recommended_tracks

# Set display option to show all columns
pd.set_option('display.max_columns', None)

# recommendation and reranking paramteres
N = 5000
k = 25
p_fair = 0.98
lambd = 0.99


def create_recommendation_lists(user_id, initial_uris, ngrok_url, profile_sample_size=50, ):
    
    """
    Professes the recommendation lists by calling the model and reranking the results
    """
    
    # If no user_id is provided, use the current time alternatively (some error occured during saving the user_id in the session)
    if user_id == None:
        user_id = datetime.now().strftime("%H:%M:%S")

    # Load the precomputed data for combining Spotify with LFM and the precomputed track popularities
    spids_ids = pd.read_csv("/app/data/spotify_uris.csv")
    track_popularities = pd.read_csv("/app/data/track_popularity.csv")
    
    # get the spids of the user profile's uris
    initial_spids = uris_to_spids(initial_uris)
    del initial_uris

    # check which are valid
    valid_initial_spids = get_valid_spids(initial_spids, spids_ids)

    # if there are not enough valid spids, raise an exception
    if len(valid_initial_spids) < 5:
        raise InsufficientItemsException("Not enough items")
    
    insufficientItems = 'False'
    if len(valid_initial_spids) < 100:
        insufficientItems = 'True'
        
    # sample the valid spids to create a user profile for validation
    profile_sample = random.sample(valid_initial_spids, profile_sample_size)

    # get the ids of the valid spids
    initial_track_ids = spids_to_ids(valid_initial_spids, spids_ids)
    
    # create the user profile by computing some metrics
    user_profile = create_user_profile(user_id, initial_track_ids, track_popularities, len(initial_spids))
    print(user_profile)

    del initial_spids, valid_initial_spids 
    


    payload = {
        "user_id": user_id,
        "initial_track_ids": initial_track_ids
    }

    try:
        # Make a POST request to the Flask app running on the Ngrok URL
        response = requests.post(ngrok_url + "/get_recommendations", json=payload)

        del payload
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response from the Flask app
            data = response.json()

            # Extract the recommendations from the response
            recommendations_idxs = [data["recommendation_idxs"]]
            recommendations_ids = [data["recommendation_ids"]]
            scores = [data["scores"]]

            print('Sucessfully retrieved idxs, scores and ids')
        else:
            print("Error:", response.text)

    except requests.exceptions.RequestException as e:
        print("Error making the request:", e)

    del data
    
    
    print('Computed Recommendations')
    base_recommendations = recommendations_ids[0][:k]
    print("Created Base Recommendations")

    #Rerank the recommendations using the repsecitve algorithms
    fair_reranked_ids = rerank('FAIR', recommendations_idxs, recommendations_ids, scores, k=k, user_ids=[user_id], 
                                  user_profiles=user_profile[['user_id', 'head_ratio', 'mid_ratio', 'tail_ratio']], track_popularities=track_popularities, alpha_fair=0.1, p_fair=p_fair)
    print("Created FAIR Recommendations")

    cp_reranked_ids = rerank('CP', recommendations_idxs, recommendations_ids, scores, k=k, user_ids=[user_id], 
                                user_profiles=user_profile[['user_id', 'head_ratio', 'mid_ratio', 'tail_ratio']], track_popularities=track_popularities, delta=lambd)
    print("Created CP Recommendations")
    
    # Compute metrics about the recommendations
    recommendation_stats, similarity_stats = analyse_recommendations(user_id, base_recommendations, fair_reranked_ids[0], cp_reranked_ids[0], 
                                                                     user_profile, track_popularities)
    
    print(recommendation_stats)
    print(similarity_stats)
    try:
        # Save the metrics of the user profile and recommendation lists
        write_recommendations_stats(user_id, user_profile, recommendation_stats, similarity_stats)
    except Exception as e:
        return "Error: Logging failed."
    
    # Transform the ids to spids
    base_recommendation_uris = ids_to_spids(base_recommendations, spids_ids)
    
    FAIR_recommendation_uris = ids_to_spids(fair_reranked_ids[0], spids_ids)
    
    CP_recommendation_uris = ids_to_spids(cp_reranked_ids[0], spids_ids)
    try:
        # save the recommendation lists
        write_recommended_tracks(user_id, base_recommendation_uris, FAIR_recommendation_uris, CP_recommendation_uris)
    except Exception as e:
        return "Error: Logging failed."
    
    return base_recommendation_uris, FAIR_recommendation_uris, CP_recommendation_uris, profile_sample, insufficientItems
    

class InsufficientItemsException(Exception):
    pass

class LoggingErrorException(Exception):
    pass