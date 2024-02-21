import os
import sys
import requests
from flask import Flask, request, jsonify
from flask_session import Session
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from LFMRecommendations.recommendation_handler import get_base_recommendations

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

ngrok_url = sys.argv[0]

def keep_ngrok_alive(ngrok_url):
    # Continuously ping the Ngrok URL to keep the tunnel alive
    while True:
        try:
            requests.get(ngrok_url)
        except requests.exceptions.RequestException:
            pass
        
        
@app.route("/get_recommendations", methods=["POST"])
def get_recommendations():
    try:
        
        user_id = request.json.get("user_id")
        initial_track_ids = request.json.get("initial_track_ids")

        # Call your recommendation function and get the results
        recommendations_idxs, recommendations_ids, scores = get_base_recommendations(user_id, initial_track_ids)
        
        # Convert the NumPy arrays to Python lists for JSON serialization
        recommendations_idxs = [int(entry) for sublist in recommendations_idxs for entry in sublist]

        recommendations_ids = [int(entry) for sublist in recommendations_ids for entry in sublist]


        scores =  [float(entry) for sublist in scores for entry in sublist]



        # Prepare the response
        response = {
            "recommendation_idxs": recommendations_idxs,
            "recommendation_ids": recommendations_ids,
            "scores": scores
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Start the Ngrok ping thread
    ping_thread = threading.Thread(target=keep_ngrok_alive, args=(ngrok_url,))
    ping_thread.daemon = True
    ping_thread.start()

    # Start the Flask app
    app.run(debug=True)
