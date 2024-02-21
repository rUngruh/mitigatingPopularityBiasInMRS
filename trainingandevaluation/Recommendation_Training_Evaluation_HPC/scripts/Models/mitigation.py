# -*- coding: utf-8 -*-

import numpy as np
import math
import pandas as pd
from fairsearchcore.models import FairScoreDoc

def rerank_XQ(initial_list, scores, track_popularities, user_profile, delta=0, k=None):
    """
    Apply reranking algorithm XQ to the initial recommendations
    https://arxiv.org/pdf/1901.07555.pdf
    """
    reranked_list = []
    
    # save counts of each category
    category_counts = {'tail' : 0,
                       'mid': 0,
                       'head': 0}
    
    # if no k provided, rerank the entire list
    if k==None:
        k = len(initial_list)
    
    # get the popularity of each item in the initial list
    item_popularities = [track_popularities[track_popularities['track_id']==item]['popularity'].item() for item in initial_list]

    # iterate and select and add the item with the highest criterion
    for i in range(k):
        criterion = [(1 - delta) * score + delta * marginal_likelihood(item, item_popularity, initial_list, user_profile, category_counts, len(reranked_list)) for item, item_popularity, score  in zip(initial_list, item_popularities, scores)]
        
        selected_idx = np.array(criterion).argmax()
        
        reranked_list.append(initial_list[selected_idx])
        category_counts[item_popularities[selected_idx]] += 1
        
        del initial_list[selected_idx]
        del scores[selected_idx]
        del item_popularities[selected_idx]

    return reranked_list

def marginal_likelihood(item, item_popularity, remaining_list, user_profile, category_counts, list_len):
    """
    Computes the marginal likelihood, the criterion for XQ
    """
    if list_len == 0:
        return 0
    
    score = 0

    user_profile_copy = user_profile.copy()
    user_profile_copy['head_mid_ratio'] = user_profile_copy['mid_ratio'] + user_profile_copy['head_ratio']
    
    category_counts_copy = category_counts.copy()
    category_counts_copy['head_mid'] = category_counts['head'] + category_counts['mid']
    
    if item_popularity == 'head' or item_popularity == 'mid':
        item_pop = 'head_mid'
    else:
        item_pop = item_popularity
    
    for c in ['head_mid', 'tail']:
        if item_pop != c:
            continue
        else:
            
            score += (user_profile_copy[c + '_ratio'] * math.prod([1 - (category_counts_copy[c] / list_len)]))
    return score


def rerank_CP(initial_list, scores, track_popularities, user_profile, delta=0, k=None):
    """
    Re-rank the initial recommendations using the CP algorithm
    https://arxiv.org/abs/2103.06364
    """
    reranked_list = []
    
    # save counts of each category
    category_counts = {'tail' : 0,
                       'mid': 0,
                       'head': 0}
    
    score_count = 0
    
    if k==None:
        k = len(initial_list)

    # get the popularity of each item in the initial list
    item_popularities = [track_popularities[track_popularities['track_id']==item]['popularity'].item() for item in initial_list]
    
    # iterate and select and add the item with the highest criterion
    for i in range(k):
        criterion = marginal_relevances(score_count, scores, item_popularities, category_counts, len(reranked_list), user_profile, delta)
        
        selected_idx = np.array(criterion).argmax()

        score_count += scores[selected_idx]
        
        reranked_list.append(initial_list[selected_idx])
        category_counts[item_popularities[selected_idx]] += 1
        
        del initial_list[selected_idx]
        del scores[selected_idx]
        del item_popularities[selected_idx]
    return reranked_list

def marginal_relevances(score_count, item_scores, item_popularities, category_counts, list_len, user_profile, delta):
    """
    Computes the marginal relevance, the criterion for CP
    """
    relevances = np.zeros(len(item_scores))
    recommendation_counts = pd.DataFrame({'head_ratio' : [category_counts['head']],
                             'mid_ratio' : [category_counts['mid']],
                             'tail_ratio' : [category_counts['tail']]})
    computed_categories = set()

    for i, (score, popularity) in enumerate(zip(item_scores, item_popularities)):
        if popularity in computed_categories:
            continue

        recommendation_counts[popularity + '_ratio'] += 1
        recommendation_ratios =recommendation_counts / (list_len + 1)
        
        relevances[i] = ((1-delta) * (score_count + score) - delta * jensen_shannon(recommendation_ratios, user_profile))
        recommendation_counts[popularity + '_ratio'] -= 1

        computed_categories.add(popularity)

    return relevances
        
    
def jensen_shannon(recommendation_ratios, user_profile):
    """
    Jensen Shannon divergence between the recommendation ratios and the user profile
    """
    epsilon = 1e-8  # Small non-zero value
    
    A = 0
    B = 0

    
    for c in ['head_ratio', 'mid_ratio', 'tail_ratio']:
        
        profile_ratio = user_profile[c].item()
        
        recommended_ratio = recommendation_ratios[c].item()
        
        if profile_ratio == 0:
            profile_ratio += epsilon
        
        if recommended_ratio == 0:
            recommended_ratio += epsilon
    
        A += profile_ratio * math.log2((2 * profile_ratio) / (profile_ratio + recommended_ratio))
        B += recommended_ratio * math.log2((2 * recommended_ratio) / (profile_ratio + recommended_ratio))
        
    js = (A + B) / 2
    return js


def rerank_fair(initial_list, scores, track_popularities, fair):
    """
    Re-rank the initial recommendations using the FA*IR algorithm
    https://dl.acm.org/doi/abs/10.1145/3132847.3132938?casa_token=nzSJG6Wb0f8AAAAA:mrqMkrujnjy3X-shIBWVDWD6fTJMs6fkJRNd72-GvNrwWkTnP8rgRFuPs7TJYnIlMDJNDbyYlvQ
    """
    item_popularities = [track_popularities[track_popularities['track_id']==item]['popularity'].item() for item in initial_list]
    
    item_protection = [False if popularity == 'head' or popularity == 'mid' else True for popularity in item_popularities]
    
    unfair_ranking = [FairScoreDoc(iid, sc, ip) for iid, sc, ip in zip(initial_list, scores, item_protection)]
    
    re_ranked = fair.re_rank(unfair_ranking)

    reranked_list = [fsd.id for fsd in re_ranked]


    
    return reranked_list