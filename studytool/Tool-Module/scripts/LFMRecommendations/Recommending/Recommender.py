# -*- coding: utf-8 -*-

    
def predict(model, user_ids=[], user_idxs=[], user_items=[], N=10, user_index_map={}, track_index_map={}):
    """
    predicts recommendations for a list of user_ids given a trained model
    """
    
    if user_ids != []:
        # transform the user_ids to user_idxs that correspond to the index in the user_items matrix
        user_idxs =  [idx for idx, user_id in user_index_map.items() if user_id in user_ids]
    
    if user_idxs != []:
        # predict recommendations for the user_idxs and transform the idxs back to ids
        recommendations_idxs, scores = model.predict(user_idxs, N, user_items)
        recommendations_ids = [[track_index_map[index] for index in recommended_indices] for recommended_indices in recommendations_idxs]
        
        return recommendations_idxs, recommendations_ids, scores
    else:
        print('no fitting indices found')
        return [], [], []
