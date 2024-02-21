# -*- coding: utf-8 -*-

import pandas as pd
import random

def get_personalized_recommendations(sp, top_tracks, iterations=10):
    """
    Get personalized recommendations from the Spotify API based on the user's top tracks
    """
    
    # Extract the necessary information from the recommendation response
    recommended_tracks = []
    for trial in range(iterations):
        random.shuffle(top_tracks)
        print("Trial:", trial)
        for track_num in range(0, len(top_tracks), 5):

            # Extract the track IDs from the top tracks
            seed_tracks = top_tracks[track_num:track_num+5]
            
            # Get personalized recommendations based on the user's seed tracks
            recommendations = sp.recommendations(limit=100, seed_tracks=seed_tracks)
            
            for track in recommendations['tracks']:
                recommended_tracks.append(track['uri'])
    
    return recommended_tracks