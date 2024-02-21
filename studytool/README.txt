Code for the study tool used in the work: ### add paper ###

This module combines a collaborative filtering algorithm, namely RankALS, and various reranking algorithms (CP, XQ, and FA*IR) with user profiles extracted from the Spotify API.
Using this, a study investigating the impacts of popularity bias mitigation on the user's perception can be created and evaluated.

This code can easily be adapted to be used with other recommendation and mitigation algorithms.

The tool directories work together as two modules, where Tool-Module handles the study tool and the processing of the Spotify API, as well as, the re-ranking. 
The LocalModelHost handles the trained recommendation algorithm on a server. For more information, read the readme's of the respective modules.

This code does not include the needed data and trained models. Those can be replicated using the code from: ### add git with code for training ###
Please observe both modules readme's there to generate the needed data.
