This module includes the full model used for the study. This was deployed to Heroku and only works in combination with the LocalModelHost. 
The module handles the displaying of templates, the saving of interaction data by sending emails.
By deploying it to Heroku, you will be able to run the study yourself. In scripts/LFMRecommendations/recommendation_handler.py, the recommendations are created and returned to the application. 
For creating different types of recommendations consider changing the computations done in this script and its imports. 
Additionally, you will need to add the questionnaires. Change their urls in the credentials.env or Config Vars on Heroku.

In a future version consider improving and fixing the error messages. Currently they are not optimally implemented in the worker.
This module makes use of a redis worker for computing the recommendations in the background. This is necessary because the recommendations will load for more than 30 seconds which is the limit for Heroku. 
Using Redis is also possible on a local server. I have never tested this. For using Redis locally, you would also need to add variables in the credentials.
Additionally, this tool needs the LocalModelHost to be running on a computer continuously. It is possible to make use of services like Microsoft Azure. 
I have tested this, but for the purpose of this work, it was more convenient to use the local server with NGROK.

Add your credentials, email credentials and ngrok url in the credentials.env or to the config vars on Heroku.


Making this script running properly on Heroku was a quite extensive process. I will provide a description of the steps as thorough as possible. 

For deploying and running the module accomplish the following steps:
0. Set your email address and secret in the scripts\Application\logging.py
1. Upload this directory to git. Make sure, the requirements.txt and runtime.txt are up to date.
2. Create a Heroku account. Heroku provides free credits for students that have a GitHub Student Developer account. Activate your credits at https://www.heroku.com/students
3. Create a Heroku project.
4. Go to the Deploy tab. Deploy this project using git. (Manual deploys using Github was the most convenient for me)
5. If the requirments.txt, the runtime.txt and the Procfile were properly defined, Heroku should automatically set the web app and application. 
	(requirements2.txt was defined at a later stage but should work as well while using less packages, needs testing)
    Under Resources>Basic Dynos, you should see "web gunicorn stripts.app:app" and "worker python scripts/worker.py" 
	(Heroku might struggle with the python version. Since I used python 3.8, I had to use the heroku-20 stack. Consider testing the app with a higher version of python and using the most recent stack)
6. Additionally, you should add the redis "Heroku Data for Redis" add on. Basic dynos and the Mini add-on worked without any issues for me.
7. Set the config variables under Settings>Config Vars. Reveal the Config Vars. The Redis URLs should've been set automatically. 
	Add your Spotify credentials and the links provided as provided in the config vars (For more information on how to set up the SpotifyAPI see below). Also add your questionnaire URLs.
8. Start the LocalModelHost on a local computer and keep it running at all time. For more information see the readme in the LocalModelHost directory.
9. Test the application by clicking "Open App" I would recommend observing the logs as well ("More>ViewLogs"). 
	You will see your python print statements as well as the status of the worker etc. This is pretty helpful for debugging.
10. Users can access the tool by the domain provided under Settings>Domains


For creating the Spotify application, some details have to be considered. Please follow these step-by-step instructions
1. Create a Spotify Developers account. I used my private one, but you can also create a free account only for the development: https://developer.spotify.com/
2. Go to your Dashboard and Create and application "Create app"
3. Go to Settings
4. Add your Client ID and client Secret to the credentials.env of your app or to your Config Vars on Heroku if your app is running on Heroku
5. Add the Redirect URIs. Add "http://127.0.0.1:5000/callback" for access on your local computer. 
	For access on Heroku you need your app domain as found on Heroku under Settings>Domains. 
	Add "<your Domain>/callback". Add the callback domain to your credentials.env or your Heroku Config Vars respectively.
6. Add users to your application. Under Settings>User Management you can define access for users. 
	Get the Spotify E-Mail addresses of your users and add them. This way, your users will be able to use the tool.


When you are ready to conduct the study, you'll need to consider some things. These step-by-step instructions helped me to keep track of everything important.
A: Before the study
1. Deploy the Application to Heroku (see above) and create an application in Spotify for Developers
2. Make sure the tool is running properly. Test it yourself and conduct pre-tests with users. Remember to add them as users in the Spotify API
3. Create Questionnaires. I created four questionnaires:
	a. Initial Questionnaire: Users are able to provide a broad time frame in which they want to conduct the study. 
	This is not necessarily needed, but since you can add a maximum of 25 users in the Spotify API it might make sense to limit the time in which users can participate. 
	Also inquire the users Spotify Email address as well as an email address to contact them.
	b. Pre Questionnaire: Initial questions about the users. Inquire demographics and other user factors (like musical engagement)
	c. Post Questionnaire: Questionnaires about the users perception of the individual recommendation lists
	d. Final Questionnaire: Questionnaires about the profile validation and final remarks. Also ask for the user's email again to revoke the access to the study tool
4. Send invitations for the study with a link to the initial questionnaires.
5. Keep the tool on Heroku and the LocalModelHost running at all time. 
6. I experienced that the LocalModelHost sometimes did not work. Make sure to restart it from time to time and if you restart the ngrok server, make sure to add the new url to the config vars.

B: During the study
1. Observe when users sign up for conducting the study. On the selected day add their email addresses to the Spotify API (Settings>User management)
2. Send Emails to the users informing them that you enabled their access. Send them the link to the study.
3. Let users conduct the experiment by themself. Ensure that you are available as much as possible if questions occur.
4. Observe the results of the final questionnaire. If users are done with the experiment revoke their access by deleting them from the Spotify Developers dashboard.




Remark: Please add the data to the correct paths, see:
- Study-Tool\LocalModelHost\data\Add_missing_files.txt
- Study-Tool\LocalModelHost\Models\Add_missing_files.txt
- Study-Tool\Tool-Module\data\Add_missing_files.txt