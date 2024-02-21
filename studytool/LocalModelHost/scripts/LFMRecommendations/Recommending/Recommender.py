# -*- coding: utf-8 -*-


    
def predict(model, user_ids=[], user_idxs=[], user_items=[], N=10, user_index_map={}, track_index_map={}):
    """
    Use the trained model to predict recommendations for a given user
    """
    
    # if the ids were provided, transform them to indices of the matrix
    if user_ids != []:
        user_idxs =  [idx for idx, user_id in user_index_map.items() if user_id in user_ids]
    
    # if indices were provided or computed, compute the recommendations
    if user_idxs != []:
        recommendations_idxs, scores = model.predict(user_idxs, N, user_items)
        recommendations_ids = [[track_index_map[index] for index in recommended_indices] for recommended_indices in recommendations_idxs]
        
        
        # return the recommendations
        return recommendations_idxs, recommendations_ids, scores
    else:
        print('no fitting indices found')
        return [], [], []
