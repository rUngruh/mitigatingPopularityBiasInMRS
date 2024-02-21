import os


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import pickle


import re

from Models.baselines import popularity_ranker, random_ranker

from Models.mitigation import rerank_CP, rerank_XQ, rerank_fair
from Models.RankALS import RankALS
from Models import baselines
import fairsearchcore as fsc



def train_and_save_model(modelName, trainset, modelPath = "", iterations=15, factors=32):
    """
    Based on the modelName and parameters, a model is trained and saved in the modelPath.
    """
    
    # select model
    if modelName == 'Random':
        model = random_ranker()
    elif modelName == 'Popularity':
        model = popularity_ranker()
    elif re.match(r'^RankALS.*', modelName):
        model = RankALS(iterations=iterations, factors=factors)
    else:
        print('No model selected')
    
    
    # Fit the model to the trainset
    print('Training model...' +  modelName)
    model.fit(trainset)
    
    # Save the model using pickle
    savePath = os.path.join(modelPath, modelName + '.pkl')
    print('Model trained, saving now...')
    with open(savePath, "wb") as f:
        pickle.dump(model, f)
    print('Model saved.')
    
def load_and_retrain_model(initModelName, newModelName, trainset, modelPath = "", savePath="", iterations = 5):
    """
    loads a trained model, trains it for more iterations on the trainset and saves it under a new name.
    """
    
    # load model
    model = load_model(modelName=initModelName, modelPath=modelPath)
    
    # fit model to trainset
    model.fit(trainset, iterations = 5)
    
    # save model using pickle
    savePathModel = os.path.join(savePath, newModelName + '.pkl')
    print('Model trained, saving now...')
    with open(savePathModel, "wb") as f:
        pickle.dump(model, f)
    print('Model saved.')
    
def load_model(modelName='', modelPath=""):
    """
    load a trained model
    """
    
    if modelName == 'Random':
        model = random_ranker()
    elif modelName == 'Popularity':
        model = popularity_ranker()
    elif re.match(r'^RankALS.*', modelName):
        model = RankALS()
    else:
        print('No model selected')
        return None
    
    print('Loading model...')
    loadPath = os.path.join(modelPath, modelName + '.pkl')
    try:
        with open(loadPath, "rb") as f:
            model = pickle.load(f)
        print('Model loaded.')
        return model
    except FileNotFoundError:
        print('Model file not found.')
        return None
    
    
def predict(model, user_ids=[], user_idxs=[], user_items=[], N=10, user_index_map={}, track_index_map={}):
    """
    Uses a trained model to predict the top N recommendations for a users
    """
    if user_ids != []:
        # if indices are not given, find the indices of the user_ids
        user_idxs =  [idx for idx, user_id in user_index_map.items() if user_id in user_ids]
    
    if user_idxs != []:
        # call the predict function of the model to predict new items for the users
        recommendations_idxs, scores = model.predict(user_idxs, N, user_items)
        
        # transform item idxs into ids
        recommendations_ids = [[track_index_map[index] for index in recommended_indices] for recommended_indices in recommendations_idxs]

        return recommendations_idxs, recommendations_ids, scores
    else:
        print('no fitting indices found')
        return [], [], []


def rerank(algorithm, initial_idxs, initial_ids, initial_scores, k=10, user_ids=[], user_profiles=None, track_popularities=None, delta=0, alpha_fair=0.1, p_fair=0.5):
    """
    Uses initial recommendations and scores to compute new recommendations and scores based on the given reranking algorithm
    """
    
    
    print(f'reranking now, with delta={delta}...')
    reranked_ids = []
    reranked_idxs = []
    reranked_scores = []
    
    # if FAIR, create the Fair object from fairsearchcore
    if algorithm == 'FAIR':
        f_adjusted = fsc.Fair(k, p_fair, alpha_fair)
    
    # for each user, call the reranking algorithm    
    for user_id, ids, idxs, scores in zip(user_ids, initial_ids, initial_idxs, initial_scores):
        
        scores = list(scores)
        user_profile = user_profiles[user_profiles['user_id']==user_id]
        
        # call the respective reranking function
        if algorithm == 'XQ':
            reranked_list = rerank_XQ(ids[:], scores[:], track_popularities, user_profile, delta=delta, k=k)
            
        elif algorithm == 'CP':
            reranked_list = rerank_CP(ids[:], scores[:], track_popularities, user_profile, delta=delta, k=k)
            
        elif algorithm == 'FAIR':
            reranked_list = rerank_fair(ids[:], scores[:], track_popularities, f_adjusted)
            
        reranked_ids.append(reranked_list)
        
        #Rerank the initial_idxs and initial_scores based on the sorted_ids
        reranked_idxs.append([idxs[ids.index(id_)] for id_ in reranked_list])
        reranked_scores.append([scores[ids.index(id_)] for id_ in reranked_list])
        
    print('reranked!')
    
    return reranked_ids, reranked_idxs, reranked_scores
