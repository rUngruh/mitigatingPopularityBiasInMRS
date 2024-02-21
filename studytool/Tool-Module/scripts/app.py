# -*- coding: utf-8 -*-


"""
This is the main application of the project. It contains the Flask app that is used to run the user study.
"""


import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


#lots of imports

import secrets
from flask import Flask, request, render_template, redirect, session, url_for
from flask_session import Session

import json
import os
from dotenv import load_dotenv

import datetime
import random
import string
import pandas as pd

import spotipy
from requests.auth import HTTPBasicAuth
import requests
from rq import Queue
from worker import conn
from rq.job import Job
import time

# not used anymore

#import csv
#import numpy as np
#import subprocess
#import webbrowser


from Application.SpotifyDataProcessing.Extraction import get_top_tracks
from Application.SpotifyDataProcessing.Recommendation import get_personalized_recommendations
from LFMRecommendations.recommendation_handler import create_recommendation_lists, InsufficientItemsException
from LFMRecommendations.TrackProcessing.TrackAnalysis import analyse_interaction_data
from Application.SpotifyDataProcessing.URIHandler import uri_to_track
from Application.logging import write_interaction_data, write_fails

# Create app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

app.config['SESSION_COOKIE_SECURE'] = True

# get environment variables
dotenv_path = "../credentials.env"

load_dotenv(dotenv_path=dotenv_path)

ngrok_url = os.getenv("NGROK_URL")
redirect_uri = os.getenv("redirect_uri")
token_url = os.getenv("token_url")
auth_url = os.getenv("auth_url")
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
scope = os.getenv("scope")

sp_oauth = spotipy.oauth2.SpotifyOAuth(client_id=client_id, client_secret=client_secret,
                                      redirect_uri=redirect_uri, scope=scope)

# add temporary for saving the recommended tracks
tmp_directory = "/app/tmp"
os.makedirs(tmp_directory, exist_ok=True)


# Define the directory where the recommendation files will be stored
recommendation_directory = tmp_directory #'/app/RecommendationLog' as an alternative directory

log_directory = tmp_directory #'/app/Logs' as an alternative directory

# get urls to the surveys
postquestionnaire_url = os.getenv("postquestionnaire_url")
prequestionnaire_url = os.getenv("prequestionnaire_url")
finalquestionnaire_url = os.getenv("finalquestionnaire_url")

# Create a queue for computing recommendation jobs
q = Queue(connection=conn)


def get_headers(token):
    return {"Authorization": "Bearer " + token}


# Starting point of the application
@app.route('/')
def index():
    # Generate a random ID for the user
    user_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    print(user_id)
    # Store the user ID in the session
    session['user_id'] = user_id
    
    # Render the login.html template
    return render_template('login.html', user_id=user_id)

# login route, redirects to spotify login
@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

# callback route, after Spotify login, redirects to loading screen
@app.route('/callback')
def callback():
    code = request.args.get('code')

    # Make a request to the Spotify API to get the access token
    resp = requests.post(token_url,
                         auth=HTTPBasicAuth(client_id, client_secret),
                         data={
                             'grant_type': 'authorization_code',
                             'code': code,
                             'redirect_uri': redirect_uri
                         })
    refresh_token = resp.json()['refresh_token']
    access_token = resp.json()['access_token']
    
    expires_in = resp.json()['expires_in']
    
    if access_token:
        
        # Get the refresh token
        expiration_time = time.time() + expires_in
        session['refresh_token'] = refresh_token
        session['expiration_time'] = expiration_time
        
        # Store the access token in the session
        session['access_token'] = access_token
        user_id = session.get('user_id')
        
        # Create a new job to compute the recommendations
        job = q.enqueue(do_compute_recommendations, user_id, access_token)
        print(job)

        # access the job id
        job_id = job.get_id()
        session['job_id'] = job_id
        
        # render the loading screen
        return render_template('loading.html', user_id=user_id, job_id=job_id, access_token=access_token, questionnaire_url=prequestionnaire_url)
    return "Error: Unable to authenticate with Spotify. If this error persists, please contact the researcher: Researcher Name (research@mail.com)."



def do_compute_recommendations(user_id, access_token):
    try:
        # Create a new Spotipy client with the access token
        sp = spotipy.Spotify(auth=access_token)

        # Get the current user's preferences
        long_term_tracks = get_top_tracks(sp)

        if len(long_term_tracks) <25: 
            return "Couldn't retrieve appropriate items from your user profile."

        iterations = 10
        # Compute the recommendations which will serve as the user profile
        recommended_tracks = get_personalized_recommendations(sp, long_term_tracks, iterations)
        
        # Compute the recommendations using the algorithm trained on the LFM dataset
        try:
            base_recommendation_uris, FAIR_recommendation_uris, CP_recommendation_uris, profile_sample, insufficientItems = \
                create_recommendation_lists(user_id, recommended_tracks, ngrok_url, profile_sample_size=50)
        except InsufficientItemsException as e:
            return redirect(url_for('failed_recommendation_screen'))
        

        # Convert the URIs to track objects
        final_base_recommendations = uri_to_track(sp, base_recommendation_uris)
        final_fair_recommendations = uri_to_track(sp, FAIR_recommendation_uris)
        final_cp_recommendations = uri_to_track(sp, CP_recommendation_uris)
        final_profile_sample = uri_to_track(sp, profile_sample)

        # return the recommendations
        return final_base_recommendations, final_fair_recommendations, final_cp_recommendations, final_profile_sample, insufficientItems

        
    except Exception as e:
        return {'error': 'An error occurred while processing the recommendations.'}

# loading screen, checks if the job is finished
@app.route('/check_status')
def check_status():
    user_id = request.args.get('user_id')
    job_id = request.args.get('job_id')
    job = Job.fetch(job_id, connection=conn)
    if job.is_finished:
        return "completed"
    return "pending"

# instruction screen, redirects to the resepctive recommendation screen
@app.route('/instruction_screen/<int:index>')
def instruction_screen(index):
    insufficientItems = 'False' # Set initial value for insufficientItems
    
    # Check if the index is 0, which means that the user is on the first recommendation screen
    if index == 0:
        
        # randomize the order of the recommendation lists
        recommendation_order = ['base', 'fair', 'cp']
        random.shuffle(recommendation_order)
        
        # Store the recommendation order in the session
        session['recommendation_order'] = recommendation_order
        user_id = session.get('user_id')
        
        # Get the job ID from the session
        job_id = session.get('job_id')
        job = Job.fetch(job_id, connection=conn)
        results_tuple = job.return_value()
        
        # Check if the job was successful
        if len(results_tuple) != 5:
            return str(results_tuple) + "\nPlease try again. If it still doesn't work, contact the researcher: Researcher Name (research@mail.com)"
        
        
        else:
            # get the recommendations from the job
            final_base_recommendations, final_fair_recommendations, final_cp_recommendations, final_profile_sample, insufficientItems  = results_tuple
            # Generate unique filenames for each user
            base_filename = f"{user_id}_base_recommendations.json"
            fair_filename = f"{user_id}_fair_recommendations.json"
            cp_filename = f"{user_id}_cp_recommendations.json"
            profile_filename = f"{user_id}_profile_recommendations.json"

            # Store the recommendations in the recommendation directory
            with open(os.path.join(recommendation_directory, base_filename), 'w') as f:
                json.dump(final_base_recommendations, f)
            
            with open(os.path.join(recommendation_directory, fair_filename), 'w') as f:
                json.dump(final_fair_recommendations, f)
            
            with open(os.path.join(recommendation_directory, cp_filename), 'w') as f:
                json.dump(final_cp_recommendations, f)

            with open(os.path.join(recommendation_directory, profile_filename), 'w') as f:
                json.dump(final_profile_sample, f)

    else:
        # Get the recommendation order and user_id from the session
        recommendation_order = session.get('recommendation_order')
    
        user_id = session.get('user_id')
    
    # set condition
    if index == 0:
        condition = 'pre'
    elif index < len(recommendation_order)+1:
        condition = recommendation_order[index-1]
    else:
        condition = 'Profile_validation'
        
    # Check whether a trial was completed
    if index > 0 and index < len(recommendation_order)+1:
        
        # get the data from the user's interaction
        time_spent = request.args.get('time')
        track_uris = request.args.get('uris')
        ranks = request.args.get('ranks')
        
        # Save the data to a CSV file
        # Process interaction data
        time_stats, track_choices= analyse_interaction_data(user_id, condition, time_spent, track_uris, ranks)

        # Save interaction data
        write_interaction_data(user_id, index, time_stats, track_choices)

    # Check if the index is within the valid range
    if index >= 0 and index < len(recommendation_order)+1:
        # Get the questionnaire URL
        questionnaire_url = postquestionnaire_url if index > 0 else prequestionnaire_url
        return render_template('instruction_screen.html', index=index, insufficientItems=insufficientItems, questionnaire_url=questionnaire_url, user_id=user_id, condition=condition)
    
    # Redirect to the logout page if the index is out of range
    return render_template('finish.html', questionnaire_url=finalquestionnaire_url, user_id=user_id)



# personalized recommendations, the heart of the recommendation tool and the main time of the user to interact with the recommendation lists
@app.route('/personalized_recommendations/<int:index>')
def personalized_recommendations(index):
    user_id = session.get('user_id')
    recommendation_order = session.get('recommendation_order')
    
    # Get the fitting recommendation list and render the template
    if index >= 0 and index < len(recommendation_order):
        list_filename = f"{user_id}_{recommendation_order[index]}_recommendations.json"
        recommendations_path = os.path.join(recommendation_directory, list_filename)
        with open(recommendations_path, 'r') as f:
            recommendation_list = json.load(f)
        print(recommendation_order[index])
        return render_template('personalized_recommendations.html', recommended_tracks=recommendation_list, index=index)
    
    # Redirect to the profile validation screen if the index is out of range (all trials completed)
    if index == len(recommendation_order):
        list_filename = f"{user_id}_profile_recommendations.json"
        recommendations_path = os.path.join(recommendation_directory, list_filename)
        with open(recommendations_path, 'r') as f:
            recommendation_list = json.load(f)
        
        return render_template('Profile_sample.html', recommended_tracks=recommendation_list, index=index)
    
    # Redirect to the logout page if the index is out of range (after profile validation)
    return redirect(url_for('finish'))

# finish, enables user saving their playlist
@app.route('/finish')
def finish():

    return render_template('finish.html')

    
# handle fails
@app.route('/failed_recommendation_screen/<string:e>')
def failed_recommendation_screen(e):
    # Render the logout.html template
    user_id = session.get('user_id')
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    fail_stats = pd.DataFrame({'user_id': [user_id],
                               'interaction_time': [current_datetime_str],
                               'exception':[e],
                               })
    write_fails(fail_stats)
    
    return render_template('fail.html')


# handle saving of playlist    
@app.route('/save_playlist')
def save_playlist():

    user_id=session.get('user_id')
    print(user_id)
    access_token = session.get('access_token')
    
    # Get refresh token and expiration time from the session
    expiration_time = session.get('expiration_time')
    current_time = time.time()
    
    # Check if the access token is still valid
    if current_time < expiration_time:
        print("Access token is valid")
    else:
        print("Access token has expired")
        #refresh the token
        refresh_token = session.get('refresh_token')
        token_info = sp_oauth.refresh_access_token(refresh_token) 
        
        access_token = token_info['access_token']
        
    # Create a new Spotipy client with the access token
    sp = spotipy.Spotify(auth=access_token)

    # Get the current user's information
    user_info = sp.current_user()

    # Retrieve the user_id
    spotify_id = user_info['id']

    # Create a new playlist
    playlist = sp.user_playlist_create(user=spotify_id, name='User Study Playlist', description='Playlist consisting of songs that were recommended during the user study regarding Mitigation of Popularity Bias')

    playlist_id = playlist['id']


    # retrieve the recommendation lists
    song_uris = []
    recommendation_order = session.get('recommendation_order')
    for rec_category in recommendation_order:
        list_filename = f"{user_id}_{rec_category}_recommendations.json"
        recommendations_path = os.path.join(recommendation_directory, list_filename)
        with open(recommendations_path, 'r') as f:
            recommendation_list = json.load(f)
        
        for recommendation in recommendation_list:
            song_uris.append(recommendation['uri'])

    # Add the songs to the playlist
    sp.user_playlist_add_tracks(user=spotify_id, playlist_id=playlist_id, tracks=song_uris)
    
    # Clear the session by removing all session variables
    session.clear()

    # Revoke the access token by making a request to the Spotify API
    if 'access_token' in session:
        access_token = session['access_token']
        sp = spotipy.Spotify(auth=access_token)
        sp.auth_manager.revoke_access_token(access_token)

    # Render the last template
    return render_template('saved_playlist.html')

# logout
@app.route('/logout')
def logout():
    # Clear the session by removing all session variables
    session.clear()

    # Revoke the access token by making a request to the Spotify API
    if 'access_token' in session:
        access_token = session['access_token']
        sp = spotipy.Spotify(auth=access_token)
        sp.auth_manager.revoke_access_token(access_token)

    return render_template('logout.html')

    sys.exit()
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)