# -*- coding: utf-8 -*-

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
abspath = os.path.abspath(__file__)

import pandas as pd
import main

import seaborn as sns
import matplotlib.pyplot as plt





def analyse_track_popularity(dataPath = main.trackPopularityDataPath, savePath = ('Data_Analysis/' + 'popularity_distribution_reduced.png')):
    """
    Analyse the track popularity dataset and
    create a line plot to visualize the distribution of track popularity
    in the entire dataset
    """
    
    # Load the created dataset
    print("Loading the track popularity dataset...")
    track_popularity = pd.read_csv(dataPath)
    print("Dataset loaded successfully.")
        
    # Sort the tracks based on the number of interactions
    print("Sorting the tracks based on the number of interactions...")
    track_popularity = track_popularity.sort_values('interactions', ascending=False).reset_index(drop=True)
    
    # Calculate the total number of interactions
    print("Calculating the total number of interactions...")
    total_interactions = track_popularity['interactions'].sum()
    
    # Calculate the number of interactions corresponding to 20% of the total interactions
    print("Calculating the number of interactions corresponding to 20% of the total interactions...")
    interaction_cutoff = total_interactions * 0.2
    
    # Find the index where the cumulative sum exceeds the interaction cutoff
    print("Finding the index where the cumulative sum exceeds the interaction cutoff...")
    head_index = track_popularity['interactions'].cumsum().ge(interaction_cutoff).idxmax()
    
    # Find the index where the cumulative sum reaches or exceeds the interaction cutoff
    print("Finding the index where the cumulative sum reaches or exceeds the interaction cutoff...")
    tail_index = track_popularity['interactions'].cumsum().ge(total_interactions - interaction_cutoff).idxmax()
    
    # Calculate the number of head, tail, and mid items
    num_head_items = head_index + 1
    num_tail_items = len(track_popularity) - tail_index
    num_mid_items = len(track_popularity) - num_head_items - num_tail_items
    
    # Calculate the percentage of head, tail, and mid items
    total_items = len(track_popularity)
    percentage_head_items = (num_head_items / total_items) * 100
    percentage_tail_items = (num_tail_items / total_items) * 100
    percentage_mid_items = (num_mid_items / total_items) * 100
    
    # Print the results
    print(f"Number of Head Items:, {num_head_items:.4f}")
    print(f"Number of Tail Items: {num_tail_items:.4f}")
    print(f"Number of Mid Items: {num_mid_items:.4f}")
    
    print(f"Percentage of Head Items: {percentage_head_items:.4f}")
    print(f"Percentage of Tail Items: {percentage_tail_items:.4f}")
    print(f"Percentage of Mid Items: {percentage_mid_items:.4f}")
    
    # Calculate the popularity percentage for each track
    track_popularity['pop'] = (track_popularity['interactions'] / total_interactions) * 100
    
    # Plot the distribution using Seaborn
    print('Plot the distribution using Seaborn...')
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(x=track_popularity.index, y='pop', data=track_popularity, color='gray')
    ax.set_xlabel('Item Rank')
    ax.set_ylabel('Popularity (%)')
    ax.set(xticklabels=[])
    
    
    # Add vertical dotted lines for head, mid, and tail items
    ax.axvline(x=head_index, color='blue', linestyle='--', label='Head')
    ax.axvline(x=tail_index, color='red', linestyle='--', label='Tail')
    
    
    # Add symbols for head, mid, and tail items on top of the graph
    ax.text((0.5 * head_index), track_popularity['pop'].max(), 'H', color='blue', alpha=0.3, fontsize=12, ha='center')
    ax.text((head_index + tail_index) / 2, track_popularity['pop'].max(), 'M', color='green', alpha=0.3, fontsize=12, ha='center')
    ax.text((tail_index + track_popularity.index[-1]) / 2, track_popularity['pop'].max(), 'T', alpha=0.3, color='red', fontsize=12, ha='center')
    
    ax.legend()
    
    print("Saving the plot as an image file...")
    # Save the plot as an image file
    plot_path = main.plotPath + savePath
    plt.savefig(plot_path)  # Provide the desired file name and extension
    plt.show()
    

analyse_track_popularity(dataPath = main.trackPopularityDataPath, savePath = ('/Data_Analysis/' + 'Sppopularity_distribution_example.png'))        