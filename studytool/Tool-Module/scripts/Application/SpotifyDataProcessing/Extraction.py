# -*- coding: utf-8 -*-

import spotipy
import time

def get_top_tracks(sp, time_range='long_term'):
    """
    Retrieve the user's top tracks from the Spotify API
    """
    
    # Get the user's profile information
    user_info = sp.current_user()
    
    # Extract the user email
    user_email = user_info['email']
    
    # Get the user's recently played tracks
    limit = 50
    
    # Get the user's long-term top tracks
    long_term_tracks = []
    if time_range == 'currently':

        results = sp.current_user_recently_played(limit=50)
        for item in results['items']:
            track = item['track']
            long_term_tracks.append(track['uri'])
    else:

        results = sp.current_user_top_tracks(time_range=time_range, limit=limit)
        for track in results['items']:
            
            long_term_tracks.append(track['uri'])
        
    return long_term_tracks