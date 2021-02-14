#Content Based Recommendation system using genres and tags#
Course: CSE 6363 - Machine Learning

**Content-based filtering** approaches uses a series of discrete characteristics of an item in order to recommend additional items with similar properties. Content-based filtering methods are totally based on a description of the item and a profile of the user’s preferences. It recommends items based on user’s past preferences.

The **Recommendation system** takes movies that a user currently likes as input and develop **user profile** using the content (genres, tags & user ratings) of the movies. Using user profile, system calculates **weighted average score of movies not rated by user** and recommends movies based on the score.

Files included: dataPreprocessing.py, recommend.py

1. Used movielense dataset [Link](https://grouplens.org/datasets/movielens/) with 27,000,000 ratings and 1,100,000 tag applications applied to 58,000 movies by 280,000 users. Includes tag genome data with 14 million relevance scores across 1,100 tags.

2. dataPreprocessing.py - Program to clean and pre process movielense dataset, generates cleaned-dataset folder which contains movies.csv and ratings.csv files required for recommendation.

3. recommend.py - Program to recommend movies based on previous rating

Steps:
    1. Open and Run program -> Enter command - python3 recommend.py
    2. Enter user id
    3. Displays top 20 recommendation for the user