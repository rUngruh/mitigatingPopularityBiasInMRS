# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix

from LFMRecommendations.Recommending.model_handler import load_model
from LFMRecommendations.Recommending.Recommender import predict



# model parameters
N = 5000
k = 25
p_fair = 0.98
lambd = 0.99


def get_base_recommendations(user_id, initial_track_ids):
    """
    computes base recommendations, using pretrained model and using the track_ids of the user profile
    """
    
    # retrieve the necessary data
    mappings = np.load("../data/mappings.npz", allow_pickle=True)
    dec_mappings = [mappings['user_index_map_inv'].item(), mappings['track_index_map_inv'].item()]
    user_index_map, track_index_map = dec_mappings
    
    # user_index_map is not necessary for this model
    del user_index_map
    
    # load the model
    model = load_model()
    
    
    # preprocess the user profile
    user_items = lil_matrix((1, model.model.item_factors.shape[0]))
    track_index_map_reverse = {value: key for key, value in track_index_map.items()}
    
    
    # compute the item idxs of the user profile
    for tid in initial_track_ids:
        tidx =  track_index_map_reverse[tid]

        user_items[0,tidx] = 1
        
    # Convert LIL matrix to CSR format
    user_items = user_items.tocsr()
    
    # fit the user factors of the model
    user_map = model.model.partial_fit_users([user_id], user_items)
    print('Preprocessed user data')
    
    # compute the recommendations for the user based on the previously computed user factors
    recommendations_idxs, recommendations_ids, scores = predict(model, user_ids=[user_id], user_items=user_items, N=N, user_index_map=user_map, track_index_map=track_index_map)
    
    #return the recommendations
    return recommendations_idxs, recommendations_ids, scores

