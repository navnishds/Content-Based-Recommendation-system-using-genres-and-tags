import pandas as pd

class recommend():

    def recommendMovie(self, user):
        # Read from file
        movie = pd.read_csv('cleaned-dataset/movies.csv')
        rating = pd.read_csv('cleaned-dataset/ratings.csv')
        rating.drop(['count'], axis=1,inplace=True)

        # Retrieve ratings of user
        movies_rated = rating.loc[rating['userId'] == user]
        if movies_rated.empty:
            print("User does not exist. Please enter different user.")
            return
        movies_rated = movies_rated[['movieId', 'rating']]

        # Retrieve movies the user has rated greater than 3.0
        movies_rated_g3 = movies_rated.loc[movies_rated['rating']>=3.0]
        if movies_rated_g3.empty:
            movies_rated_g3 = movies_rated
        movies_rated_g3.reset_index(inplace=True)
        movies_rated_g3 = movies_rated_g3.sort_values(by='movieId', ascending=True)
        # Get movies the user has rated
        result = pd.merge(movie, movies_rated_g3, how='inner', on='movieId')
        result = result.sort_values(by='movieId', ascending=True)
        result.drop(['Unnamed: 0', 'movieId', 'title', 'rating', 'index'], axis=1,inplace=True)
        # Multiply the rating with movie tags
        result = result.mul(movies_rated_g3['rating'], axis=0)
        # Generate user profile
        weighted_genre = result.sum(axis=0, skipna=True)
        normalized_df = weighted_genre/weighted_genre.sum()
        # Get movies which are not rated by user and multiply with user profile
        notRatedMovie = movie[~movie['movieId'].isin(movies_rated.movieId)].copy()
        notRatedMovie.drop(['Unnamed: 0'],axis=1,inplace=True)
        notRatedMovie.iloc[:, 2:] = normalized_df.mul(notRatedMovie.iloc[:, 2:], axis=0)
        # Calculate weighted average for each movie and sort in decreasing order
        notRatedMovie['agg'] = notRatedMovie.iloc[:, 2:].sum(axis = 1, skipna = True)
        notRatedMovie = notRatedMovie.sort_values(by ='agg', ascending = 0)
        # Print movies
        print("Top Recommendation\n")
        print(notRatedMovie[['movieId', 'title', 'agg']].head(20))

if __name__=="__main__":
    user = int(input("Enter User ID:"))
    recommend = recommend()
    recommend.recommendMovie(user)