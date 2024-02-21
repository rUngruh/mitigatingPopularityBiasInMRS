# -*- coding: utf-8 -*-

import fairsearchcore as fsc
from LFMRecommendations.Models.mitigation import rerank_CP, rerank_XQ, rerank_fair

def rerank(algorithm, initial_idxs, initial_ids, initial_scores, k=10, user_ids=[], user_profiles=None, track_popularities=None, delta=0, alpha_fair=0.1, p_fair=0.5):
    """
    given one of the reranking algorithms, rerank the initial recommendations
    Can rerank lists of multiple users at the same time
    In the current version, initial_idxs and initial_scores are not used, but could again for future iterations of the tool
    """
    print(f'reranking now, with delta={delta}...')
    reranked_ids = []

    if algorithm == 'FAIR':
        # if fair is used, create a fairsearch object
        f_adjusted = fsc.Fair(k, p_fair, alpha_fair)
        
    for user_id, ids, idxs, scores in zip(user_ids, initial_ids, initial_idxs, initial_scores):
        # iterate over all users and their recommendations
        scores = list(scores)
        user_profile = user_profiles[user_profiles['user_id']==user_id]
        # select an algorithm and rerank the recommendations based on the criterion
        if algorithm == 'XQ':
            reranked_list = rerank_XQ(ids[:], scores[:], track_popularities, user_profile, delta=delta, k=k)
            
        elif algorithm == 'CP':
           
            reranked_list = rerank_CP(ids[:], scores[:], track_popularities, user_profile, delta=delta, k=k)
            
        elif algorithm == 'FAIR':

            reranked_list = rerank_fair(ids[:], scores[:], track_popularities, f_adjusted)
            
        reranked_ids.append(reranked_list)
        
        #Rerank the initial_idxs and initial_scores based on the sorted_ids
        
        #reranked_idxs.append([idxs[ids.index(id_)] for id_ in reranked_list])
        #reranked_scores.append([scores[ids.index(id_)] for id_ in reranked_list])
        
    print('reranked!')
    
    return reranked_ids#, reranked_idxs, reranked_scores