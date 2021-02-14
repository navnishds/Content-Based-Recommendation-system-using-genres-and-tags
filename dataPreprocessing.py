# Machine Learning Project Phase-2
# Team 5
# Topic: Content Based Recommendation system using genres and tags
# Eshan Danayakapura Jagadeesh - 1001667159
# Navnish Danayakapura Suresh - 1001753672

# Import libraries
import pandas as pd
import fuzzywuzzy
import ast

class preProcessing():

    # Open and read from csv files
    def open(self):
        self.movieDataFrame = pd.read_csv('ml-latest/movies.csv')
        self.ratingDataFrame = pd.read_csv('ml-latest/ratings.csv')
        self.genomeTagDataFrame = pd.read_csv('ml-latest/genome-tags.csv')
        self.genomeScoreDataFrame = pd.read_csv('ml-latest/genome-scores.csv')
        self.userTags = pd.read_csv('ml-latest/tags.csv')

    def removeRowsWithBlankValues(self):
        self.movieDataFrame.dropna(axis='index', how='any', inplace=True)
        self.ratingDataFrame.dropna(axis='index', how='any', inplace=True)
        self.genomeScoreDataFrame.dropna(axis='index', how='any', inplace=True)
        self.genomeTagDataFrame.dropna(axis='index', how='any', inplace=True)
        # Remove timestamp from ratings dataset
        self.ratingDataFrame.drop('timestamp', axis=1, inplace=True)

    # Select tag whose relevance score is greater than 0.80
    def selectgenomeTags(self):
        gs = self.genomeScoreDataFrame[self.genomeScoreDataFrame['relevance'] > 0.80]
        tags_scores = pd.merge(self.genomeTagDataFrame, gs, on=['tagId'])
        return(tags_scores.sort_values(['movieId'], ascending=True))

    def selectUserTags(self):
        tagcount = self.userTags.groupby('tag').agg({'tag': 'count'}).rename(columns={'tag': 'count'}).reset_index()
        self.userTags = pd.merge(self.userTags, tagcount, on=['tag'])
        self.userTags.drop(self.userTags[self.userTags['count'] < 100].index, inplace=True)
        return (self.userTags)

    # Find similar tags and combine them
    def selectTags(self, tags_scores):

        ky = tags_scores[['tag','movieId']]
        ky.drop_duplicates(keep=False,inplace=True)

        A_ky = ky.groupby('tag') \
               .agg({'movieId':'count'}) \
               .rename(columns={'movieId':'count'}) \
               .reset_index()

        # Keyword Analysis using Fuzzy Matching
        from fuzzywuzzy import process

        names_array = []
        ratio_array = []


        def match_names(wrong_names, correct_names):
            for row in wrong_names:
                x = process.extractOne(row, correct_names)
                names_array.append(x[0])
                ratio_array.append(x[1])
            return names_array, ratio_array


        df = A_ky
        wrong_names = df['tag'].dropna().values

        # Correct tags dataset
        choices_df = A_ky[A_ky['count'] > 100]
        correct_names = choices_df['tag'].values

        name_match, ratio_match = match_names(wrong_names, correct_names)

        df['correct_tag_name'] = pd.Series(name_match)
        df['tag_names_ratio'] = pd.Series(ratio_match)

        output = df[['tag','correct_tag_name','tag_names_ratio' ]]
        new_tags = output[(output.tag_names_ratio >= 81) & (output.tag_names_ratio <= 100)]
        correct_tags = pd.merge(ky[['movieId','tag']],new_tags[['tag','correct_tag_name']], on = 'tag')
        tags_clean  = correct_tags.groupby(['correct_tag_name','movieId']) \
                                   .agg({'tag':'count'}) \
                                   .rename(columns={'tag':'count'}) \
                                   .reset_index()
        tags_clean  = tags_clean.rename(columns={'correct_tag_name' : 'tag'})
        return(tags_clean)


    def oneHotEncoding(self, genomeTags, userTags):
        genomeTags.drop(['count'], axis=1, inplace=True)
        userTags.drop(['count'], axis=1, inplace=True)

        #Combine user and genome tags
        tags = genomeTags.append(userTags)
        #Filter tags - Combine similar tags
        tags = self.selectTags(tags)
        tags.drop(['count'], axis=1, inplace=True)
        #Get tags for each movie
        tags = tags.groupby('movieId').agg(lambda x: x.tolist())

        # Convert the genre column into list
        self.movieDataFrame['genres'] = self.movieDataFrame['genres'].astype(str)
        self.movieDataFrame['genres'] = self.movieDataFrame['genres'].str.split('|')

        # Add tags found for each movie to movieDataFrame and combine genres and tags
        self.movieDataFrame = pd.merge(self.movieDataFrame, tags, on = ['movieId'])
        self.movieDataFrame['genre_tag'] = self.movieDataFrame['genres'] + self.movieDataFrame['tag']
        self.movieDataFrame.drop(['genres', 'tag'], axis=1, inplace=True)
        self.movieDataFrame['genre_tag'] = self.movieDataFrame.genre_tag.astype(str).str.lower().transform(ast.literal_eval)

        #To remove duplicate
        self.movieDataFrame['genre_tag'] = self.movieDataFrame['genre_tag'].apply(set)
        self.movieDataFrame['genre_tag'] = self.movieDataFrame['genre_tag'].apply(list)

        # One-hot encoding
        dummies = pd.get_dummies(self.movieDataFrame['genre_tag'].apply(pd.Series), prefix='', prefix_sep='').sum(level=0, axis=1)
        self.movieDataFrame = pd.concat([self.movieDataFrame, dummies], axis=1)
        self.movieDataFrame.drop(['genre_tag', '(no genres listed)'], axis=1, inplace=True)
        self.movieDataFrame.drop(self.movieDataFrame.iloc[:, 2:8], inplace=True, axis=1)

    # Remove ratings of movies which was removed from movie dataset
    def removeMovieRatings(self):
        self.ratingDataFrame = pd.merge(self.movieDataFrame['movieId'], self.ratingDataFrame, on=['movieId'])

    # Remove users who have rated less than 20 movies along with their rating
    def removeUsers(self):
        # Calculate total number of ratings from each user
        userCount = self.ratingDataFrame.groupby('userId').agg({'userId': 'count'}).rename(columns={'userId': 'count'})

        # Remove users who have rated less than 20 movies
        self.ratingDataFrame = pd.merge(self.ratingDataFrame, userCount, on=['userId'])
        self.ratingDataFrame.drop(self.ratingDataFrame[self.ratingDataFrame['count'] < 20].index, inplace=True)

    def numberOfRows(self):
        print("Movie data set: ",self.movieDataFrame.shape)
        print("Rating data set: ", self.ratingDataFrame.shape)

    # Writing data frame to new csv file after cleaning and pre processing
    def writeToCsv(self):
        self.movieDataFrame.to_csv('cleaned-dataset/movies.csv')
        self.ratingDataFrame.to_csv('cleaned-dataset/ratings.csv')


if __name__=="__main__":
    data = preProcessing()
    data.open()
    # print("Number of rows in data set before data cleaning and pre processing")
    # data.numberOfRows()
    data.removeRowsWithBlankValues()
    newGenomeTags = data.selectgenomeTags()
    cleanedGenomeTags = data.selectTags(newGenomeTags)
    newUserTags = data.selectUserTags()
    cleanedUserTags = data.selectTags(newUserTags)
    data.oneHotEncoding(cleanedGenomeTags, cleanedUserTags)
    data.removeMovieRatings()
    data.removeUsers()
    # print("Number of rows in data set after cleaning and pre processing")
    # data.numberOfRows()
    data.writeToCsv()