# -*- coding: utf-8 -*-
"""
Different re-ranking algorithms with the same interface for easy comparison and usage.
Same general interface as RankALS
"""



import numpy as np

class popularity_ranker:
    """
    Simple popularity ranker. Saves the indices of the most popular tracks and returns them as recommendations.
    """
    def __init__(self):
        self.track_indices = None
        
    def fit(self, user_item_matrix):
        track_popularity = np.sum(user_item_matrix, axis=0)
        track_indices = np.argsort(-track_popularity)
        self.track_indices = track_indices
        
    def predict(self, user_idxs, n):
        #if self.track_indices == None:
        #    print('Please fit the model first to your dataset')

        return [self.track_indices[0,:n].tolist()[0] for _ in range(len(user_idxs))], []



class random_ranker:
    """
    Simple random ranker. Returns n random tracks as recommendations.
    """
    def __init__(self):
        self.num_songs = None
        
    def fit(self, user_item_matrix):
        self.num_songs = user_item_matrix.shape[1]
        
    def predict(self, user_idxs, n):
        #if self.num_songs == None:
        #    print('Please fit the model first to your dataset')
        if n > self.num_songs:
            raise ValueError("n cannot be greater than the total number of songs.")
        track_indices = []
        for i in range(len(user_idxs)):
            track_indices.append(list(np.random.choice(self.num_songs, n, replace=False)))
           
        return track_indices, []
    
