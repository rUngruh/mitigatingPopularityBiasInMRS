The module for predicting recommendations with the model. This is done to avoid loading a large model (over 1GB) to a server and using it there. The currently used model is RankALSmin which is the same as the RankALS base model but with minimal functionality. It does not include the user factors since those are not needed for partially fitting a new user. Therefore, the model is slightly smaller. The current model is loaded and saved using pkl which might not be the most appropraite approach. Future iterations should save the models attributes seperately as intended by the "implicit" module.

In the following, we provide step-by-step instructions for running this script.
1. Create a python environment including all necessary modules (see the requirements.txt and the runtime.txt)
2. Create a ngrok account. The free version without an account works as well, but the server goes offline after 2 hours. An account enables to keep the url as long as the script runs.
3. Download and install ngrok. Run ngrok
4. Log in to your ngrok account using the access token
5. Start the server: ngrok http 5000
6. Open Anaconda
7. Activate the python environment
8. Go to this directory: cd "*/LocalModelHost"
9. Run the script: python app.py
10. Insert the ngrok url to the credentials.env or to Config Variables on Heroku
11. Test it by starting the app and load the model

