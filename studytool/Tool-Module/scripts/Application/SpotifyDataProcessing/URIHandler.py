# -*- coding: utf-8 -*-


def uri_to_track(sp, uris):
    """
    Get track objects from the Spotify API for given uris
    """
    recommended_tracks = []
    
    for idx, uri in enumerate(uris):
        track = sp.track(uri)
        
        recommended_tracks.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'image': track['album']['images'][0]['url'],
                'uri': track['uri'],
                'popularity': track['popularity'],
                'rank': idx,
                'preview_url': track['preview_url'],
                'spotify_url':track['external_urls']['spotify']
            })
        
    return recommended_tracks