import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


class NearestNeighborRecommendation:
    def __init__(self, movies, ratings, top_n):
        """ This algorithm is based on KNN systems. It will return top n recommended items for input items.
        :param movies: movie-lens latest small movie datasets
        :param ratings: movie-lens latest small ratings datasets
        :param top_n: Get n recommended items
        """
        self.movies = movies
        self.ratings = ratings
        self.top_n = top_n

    def preprocessing(self):
        """ Create ratings_matrix dataframe. Fill 0 for user's nan rating data
        :return:
        """
        self.ratings = self.ratings[['userId', 'movieId', 'rating']]
        rating_movies = pd.merge(self.ratings, self.movies, on='movieId')
        self.ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title')
        self.ratings_matrix = self.ratings_matrix.fillna(0)
        return 

    def cal_item_sims(self):
        """ Calculate item similarity using cosine similarity
        :return: item silmilarity matrix
        """
        item_sims = cosine_similarity(self.ratings_matrix.transpose(), self.ratings_matrix.transpose())
        item_sims_df = pd.DataFrame(data=item_sims, index=self.ratings_matrix.columns,
                          columns=self.ratings_matrix.columns)
        return item_sims_df

    def predict_top_sim_ratings(self, item_sims_df):
        """ Predit nan ratings data by dot item-similarity matrix, raitings matrix
        :param item_sims_df:
        :return:
        """
        first_filtering = 20
        predict = np.zeros(self.ratings_matrix.values.shape)
        for col in range(self.ratings_matrix.values.shape[1]):
            top_n_items = [np.argsort(item_sims_df.values[:, col])[:-first_filtering-1:-1]]
            for row in range(self.ratings_matrix.values.shape[0]):
                predict[row, col] = item_sims_df.values[col, :][top_n_items].dot(self.ratings_matrix.values[row, :][top_n_items].T)
                predict[row, col] = predict[row, col] / np.sum(np.abs(item_sims_df.values[col, :][top_n_items]))
        self.predict = predict
        return

    @staticmethod
    def cal_mse(predicted_ratings, real_ratings):
        predicted_ratings = predicted_ratings[real_ratings.nonzero()].flatten()
        real_ratings = real_ratings[real_ratings.nonzero()].flatten()
        mse = mean_squared_error(predicted_ratings, real_ratings)
        return mse

    def get_unseen_movies(self, userId):
        user_rating = self.ratings_matrix.loc[userId, :]
        seen_movies = user_rating[user_rating>0].index.tolist()
        movie_list = self.ratings_matrix.columns.tolist()
        unseen_movies = [movie for movie in movie_list if movie not in seen_movies]
        return unseen_movies

    def get_recommend_movie(self, userId):
        unseen_movies = self.get_unseen_movies(userId)
        predict_matrix = pd.DataFrame(data=self.predict, index=self.ratings_matrix.index, columns= self.ratings_matrix.columns)
        recommend_movies = predict_matrix.loc[userId, unseen_movies].sort_values(ascending=False)[:self.top_n]
        return recommend_movies

    def run(self):
        self.preprocessing()
        item_sims_df = self.cal_item_sims()
        self.predict_top_sim_ratings(item_sims_df)
        recommend_res = self.get_recommend_movie(9)
        return recommend_res


if __name__ == '__main__':
    movies = pd.read_csv('~/ml-latest-small/movies.csv')
    ratings = pd.read_csv('~/ml-latest-small/ratings.csv')

    test = NearestNeighborRecommendation(movies=movies, ratings=ratings, top_n=10)
    print(test.run())
