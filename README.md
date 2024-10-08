# Putting Popularity Bias Mitigation to the Test: A User-Centric Evaluation in Music Recommenders
This repository includes the code and materials for the paper "Putting Popularity Bias Mitigation to the Test: A User-Centric Evaluation in Music Recommenders" by Robin Ungruh, Karlijn Dinnissen, Anja Volk, Maria Soledad Pera, and Hanna Hauptmann, published at ACM RecSys 2024.

## Abstract
Popularity bias is a prominent phenomenon in recommender systems (RS), especially in the music domain. Although popularity bias mitigation techniques are known to enhance the fairness of RS while maintaining their high performance, there is a lack of understanding regarding users’ actual perception of the suggested music. To address this gap, we conducted a user study (n=40) exploring user satisfaction and perception of personalized music recommendations generated by algorithms that explicitly mitigate popularity bias. Specifically, we investigate item-centered and user-centered bias mitigation techniques, aiming to ensure fairness for artists or users, respectively. Results show that neither mitigation technique harms the users’ satisfaction with the recommendation lists despite promoting underrepresented items. However, the item-centered mitigation technique impacts user perception; by promoting less popular items, it reduces users’ familiarity with the items. Lower familiarity evokes discovery-the feeling that the recommendations enrich the user’s taste. We demonstrate that this can ultimately lead to higher satisfaction, highlighting the potential of less-popular recommendations to improve the user experience.

## Citation
Robin Ungruh, Karlijn Dinnissen, Anja Volk, Maria Soledad Pera, and Hanna Hauptmann. 2024. Putting Popularity Bias Mitigation to the Test: A User-Centric Evaluation in Music Recommenders. In 18th ACM Conference on Recommender Systems (RecSys ’24), October 14–18, 2024, Bari, Italy. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3640457.3688102


## Information
Please refer to the ReadMe's of the two subdirectories for further information and the code for the creation of the recommender for the study and the tool developed for user interactions:

- `studytool` includes the code to the study tool that was used in the user study. It can easily be adapted for other Recommendation Algorithms and Mitigation Strategies.

- `trainingandevaluation` includes the the code that was used to do the data analysis, train the models and evaluate them as well as the mitigation algorithms.


```latex
@inproceedings{ungruh2024putting
    title = {Putting Popularity Bias Mitigation to the Test: A User-Centric Evaluation in Music Recommenders},
    author = {Ungruh, Robin and Dinnissen, Karlijn and Anja Volk and Pera, Maria Soledad and Hauptmann, Hanna},
    booktitle = {Eighteenth ACM Conference on Recommender Systems},
    year = {2024},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    series = {RecSys '24},
    doi = {10.1145/3640457.3688102},
    isbn = {979-8-4007-0505-2/24/10}
}
```