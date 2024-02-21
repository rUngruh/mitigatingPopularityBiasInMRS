# -*- coding: utf-8 -*-


import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import os
import pandas as pd
import main

import seaborn as sns
import matplotlib.pyplot as plt





abspath = os.path.abspath(__file__)

def analyse_user_profiles(dataPath = main.userProfileDataPath, savePath= ('Data_Analysis/' + 'user_group_profiles.png')):
    """
    Analyse the user profiles dataset and 
    create a stacked barplot to visualize the average ratio of tail, mid, and head items 
    for each user group
    """
    
    # Load the created dataset
    print("Loading the user profiles dataset...")
    user_types = pd.read_csv(dataPath)
    print("Dataset loaded successfully.")
    
    # Calculate the average ratio of tail, mid, and head items for each user group
    print("Calculating the average ratio of tail, mid, and head items for each user group...")
    average_ratios = user_types.groupby('user_type')[['tail_ratio', 'mid_ratio', 'head_ratio']].mean().reset_index()
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Accumulate the values for each category
    user_types['mid_head'] = user_types['head_ratio'] + user_types['mid_ratio']
    user_types['tail_mid_head'] = user_types['mid_head'] + user_types['tail_ratio']
    
    
    # Create the stacked barplot with the specified colors
    print("Creating the stacked barplot...")
    sns.barplot(x='user_type', y='tail_mid_head', data=user_types, ax=ax, label='Tail', color='red', errorbar=None)
    sns.barplot(x='user_type', y='mid_head', data=user_types, ax=ax, label='Mid', color='green', errorbar=None)
    sns.barplot(x='user_type', y='head_ratio', data=user_types, ax=ax, label='Head', color='blue', errorbar=None)
    
    
    
    # Set the labels and legend
    ax.set_xlabel("User Type")
    ax.set_ylabel("Ratio")
    ax.legend(title="Popularity")
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    )
    
    # Save the plot as an image file
    print("Saving the plot as an image file...")
    plot_path = main.plotPath + savePath
    plt.savefig(plot_path)  # Provide the desired file name and extension
    plt.show()
  
  
def analyse_profile_sizes(dataPath = main.userProfileDataPath, savePath = ('Data_Analysis/' + 'user_group_profile_sizes.png')):
    """
    Analyse the user profiles dataset and
    create a bar chart to visualize the profile size (total_items) for each user group
    """
    
    # Load the created dataset
    user_types = pd.read_csv(dataPath)
    
    plt.figure(figsize=(6, 6))
    # Create a bar chart for the profile size (total_items) for each user group
    ax = sns.barplot(x="user_type", y="total_items", data=user_types, palette=['blue', 'green', 'red'])
    ax.set_xlabel("User Type")
    ax.set_ylabel("Profile Size")
    
    # Save the plot as an image file
    plot_path = main.plotPath + savePath
    plt.savefig(plot_path)  # Provide the desired file name and extension
    plt.show()
    
    # Calculate the average profile sizes per user group
    average_profile_sizes = user_types.groupby('user_type')['total_items'].mean().reset_index()
    
    # Generate the formatted text for average profile sizes per user group
    average_sizes_text = ""
    for index, row in average_profile_sizes.iterrows():
        user_type = row['user_type']
        average_size = row['total_items']
        average_sizes_text += f"{user_type}: {average_size}\n"
    

def analyse_head_ratio(dataPath = main.userProfileDataPath, savePath = ('Data_Analysis/' + 'head_ratio.png')):
    """
    Analyse the user profiles dataset and
    create a line plot to visualize the ratio of tail items for the users in the dataset in ascending order
    """
    # Load the created dataset
    user_types = pd.read_csv(dataPath)
    user_types = user_types.sort_values('tail_ratio', ascending=True).reset_index(drop=True)
    user_types['head_ratio'] = user_types['head_ratio']*100
    user_types['mid_ratio'] = user_types['mid_ratio']*100
    user_types['tail_ratio'] = user_types['tail_ratio']*100
    
    print('Plot the distribution using Seaborn...')
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(x=user_types.index, y='tail_ratio', data=user_types, color="red", label='Ratio Tail Items')
    
    ax.set_xlabel('User')
    ax.set_ylabel('Ratio (%)')

    
    # Add vertical dotted lines for head, mid, and tail items
    ax.axhline(y=80, color="gray", linestyle='--', label='80% Ratio')

    ax.legend()
    
    print("Saving the plot as an image file...")
    # Save the plot as an image file
    plot_path = main.plotPath + savePath
    plt.savefig(plot_path)  # Provide the desired file name and extension
    plt.show()

analyse_user_profiles(dataPath = main.userProfileDataPath, savePath= ('/Data_Analysis/' + 'Spuser_group_profiles.png'))

analyse_profile_sizes(main.userProfileDataPath, ('/Data_Analysis/' + 'Spuser_group_profile_sizes.png'))

analyse_head_ratio(main.userProfileDataPath, ('/Data_Analysis/' + 'Sptail_ratio.png'))