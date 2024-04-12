# Putting Popularity Bias Mitigation to the Test: A User-Centric Evaluation in Music Recommenders

## Study Tool


This directory includes the code for the study tool used in the paper **Putting Popularity Bias Mitigation to the Test: A User-Centric Evaluation in Music Recommenders**


This module combines a collaborative filtering algorithm, namely **RankALS**, and various reranking algorithms (**CP**, **XQ**, and **FA*IR**) with user profiles extracted from the [Spotify API](https://developer.spotify.com/).
Using this, a study investigating the impacts of popularity bias mitigation on the user's perception can be created and evaluated.

This code can easily be adapted to be used with other recommendation and mitigation algorithms.

The tool directories work together as two modules, where **Tool-Module** handles the study tool and the processing of the Spotify API, as well as, the re-ranking. 
The **LocalModelHost** handles the trained recommendation algorithm on a server. For more information, read the ReadMe's of the respective modules.

This code does not include the needed data and trained models. Those can be replicated using the code from: **trainingandevaluation**, which is also included in this repository.

Please observe both modules ReadMe's in this directory to generate the needed data.
