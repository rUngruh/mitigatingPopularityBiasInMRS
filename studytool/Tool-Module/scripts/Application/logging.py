# -*- coding: utf-8 -*-
"""
Model for simplified saving of the study data, e.g. in csv files
Sends the data via email if send_mail is set to True
"""

import os
import pandas as pd

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# Set the sender and recipient email addresses
SENDER_EMAIL = 'sender_email@mail.com'
RECIPIENT_EMAIL = 'recipient_email@mail.com'

send_mail = True

def send_email_with_attachments(subject, body, attachments):
    """
    Send an email with attachments
    """
    
    # Define SMTP email server details
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587 
    EMAIL_LOGIN = 'sender_email@mail.com'
    EMAIL_PASSWORD = 'add_password'

    # Create a multipart message container
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject

    # Attach body text to the email
    msg.attach(MIMEText(body, 'plain'))

    # Attach each file as a separate attachment
    for attachment in attachments:
        with open(attachment, 'rb') as file:
            part = MIMEApplication(file.read(), Name=attachment)
            part['Content-Disposition'] = f'attachment; filename="{attachment}"'
            msg.attach(part)

    # Connect to the SMTP server and send the email
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_LOGIN, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())


# Define the names of the csv files
user_profile_path = 'user_profiles_'
recommendation_stats_path = 'recommendation_stats_'
recommendation_similarities_path = 'recommendation_similarities_'
interaction_data_path = 'interaction_data_'
time_path = 'time_data_'
track_choices_path = 'track_choices_'
choice_stats_path =  'choice_stats_'
fail_path = 'fails_'
recommendation_uris_path = 'recommendation_uris_'

def write_recommendations_stats(user_id, user_profile, recommendation_stats, recommendation_similarities):
    # Write user_profile dataframe
    user_profile_path_id = user_profile_path + user_id + '.csv'
    if os.path.exists(user_profile_path_id):
        user_profile.to_csv(user_profile_path_id, mode='a', header=False, index=False)
    else:
        user_profile.to_csv(user_profile_path_id, index=False)

    recommendation_stats_path_id = recommendation_stats_path + user_id + '.csv'
    # Write recommendation_stats dataframe
    if os.path.exists(recommendation_stats_path_id):
        
        recommendation_stats.to_csv(recommendation_stats_path_id, mode='a', header=False, index=False)
    else:
        recommendation_stats.to_csv(recommendation_stats_path_id, index=False)

    # Write recommendation_similarities dataframe
    recommendation_similarities_path_id = recommendation_similarities_path + user_id + '.csv'
    if os.path.exists(recommendation_similarities_path_id):
        # If path already exists, append the data
        recommendation_similarities.to_csv(recommendation_similarities_path_id, mode='a', header=False, index=False)
    else:
        # If no path exists, create a new file
        recommendation_similarities.to_csv(recommendation_similarities_path_id, index=False)
    if send_mail:
        # send email
        attachments = [recommendation_stats_path_id, user_profile_path_id, recommendation_similarities_path_id]
        send_email_with_attachments('Recommendations Stats ' + user_id, 'CSV files attached', attachments)

def write_interaction_data(user_id, index, time_stats, track_choices):
    time_path_id = time_path + user_id + '_' + str(index) + '.csv'
    if os.path.exists(time_path_id):
        time_stats.to_csv(time_path_id, mode='a', header=False, index=False)
    else:
        time_stats.to_csv(time_path_id, index=False)

    track_choices_path_id = track_choices_path + user_id + '_' + str(index) + '.csv'
    if os.path.exists(track_choices_path_id):
        # If path already exists, append the data
        track_choices.to_csv(track_choices_path_id, mode='a', header=False, index=False)
    else:
        # If no path exists, create a new file
        track_choices.to_csv(track_choices_path_id, index=False)
    if send_mail:
        # send email
        attachments = [time_path_id, track_choices_path_id]
        send_email_with_attachments('Interaction Data ' + user_id + '_' + str(index), 'CSV files attached', attachments)


def write_fails(fail_details):
    """
    For debugging, write the fails to a csv file
    """
    if os.path.exists(fail_path):
        fail_details.to_csv(fail_path, mode='a', header=False, index=False)
    else:
        fail_details.to_csv(fail_path, index=False)
    if send_mail:
        attachments = [fail_path]
        send_email_with_attachments('Fails Data', 'CSV file attached', attachments)

def write_recommended_tracks(user_id, base, fair, cp):
    # Write recommendation_uris dataframe
    recommendation_uris_path_id = recommendation_uris_path + user_id + '.csv'
    
    recommendation_uris = pd.DataFrame({'user_id':[user_id, user_id, user_id],
                                 'condition':['base', 'fair', 'cp'],
                                 'uris': [';'.join(base), ';'.join(fair), ';'.join(cp)],
                                 })
    if os.path.exists(recommendation_uris_path_id):
        # If path already exists, append the data
        recommendation_uris.to_csv(recommendation_uris_path_id, mode='a', header=False, index=False)
    else:
        # If no path exists, create a new file
        recommendation_uris.to_csv(recommendation_uris_path_id, index=False)
    if send_mail:
        # send email
        attachments = [recommendation_uris_path_id]
        send_email_with_attachments('Recommended Tracks ' + user_id , 'CSV file attached', attachments)