import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


class MatrixFactorizeRecommendation:
    def __init__(self, movies, ratings, userId, top_n):
        """ This algorithm is based on matrix factorization with SGD method. It returns top n recommended items for
        input user.
        :param movies: movie-lens latest small movie datasets
        :param ratings: movie-lens latest small ratings datasets
        :param userId: input userId
        :param top_n: get n recommended items
        """
        self.movies = movies
        self.ratings = ratings[['userId', 'movieId', 'rating']]
        self.userId = userId
        self.top_n = top_n
        rating_movies = pd.merge(self.ratings, self.movies, on='movieId')
        self.ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title')
        self.ratings_matrix = self.ratings_matrix.fillna(0)

    def cal_rmse(self, P, Q, real_data):
        predicted_matrix = np.dot(P, Q.T)
        i_real_indices = [real_value[0] for real_value in real_data]
        j_real_indices = [real_value[1] for real_value in real_data]
        real_ratings_matrix = self.ratings_matrix.values[i_real_indices, j_real_indices]
        real_predicted_matrix = predicted_matrix[i_real_indices, j_real_indices]

        mse = mean_squared_error(real_ratings_matrix, real_predicted_matrix)
        rmse = np.sqrt(mse)
        return rmse

    def matrix_factorization(self, steps=100, num_factors=50, learning_rate=0.01, reg_lambda=0.01):
        """ Movie Lens datasets is very sparse. Therefore use updating algorithm SGD instead of SVD.
        :param steps: steps for SGD
        :param num_factors: dimension of latent factors
        :param learning_rate:
        :param reg_lambda:
        :return:
        """
        num_users, num_movies = self.ratings_matrix.values.shape
        np.random.seed(1)
        P = np.random.normal(scale=float(1)/num_factors, size=(num_users, num_factors))
        Q = np.random.normal(scale=float(1)/num_factors, size=(num_movies, num_factors))

        real_data = [(i, j, self.ratings_matrix.values[i, j]) for i in range(num_users) for j in range(num_movies) if self.ratings_matrix.values[i, j] > 0]
        for step in range(steps):
            for i, j, r in real_data:
                error_ij = r - np.dot(P[i, :], Q[j, :].T)
                P[i, :] = P[i, :] + learning_rate*(error_ij*Q[j, :] - reg_lambda*P[i, :])
                Q[j, :] = Q[j, :] + learning_rate*(error_ij*P[i, :] - reg_lambda*Q[j, :])
            rmse = self.cal_rmse(P, Q, real_data)
            if step % 10 == 0:
                print('Iteration step ', step, ' rmse : ', rmse)
        self.predict = np.dot(P, Q.T)
        return

    def get_unseen_movies(self):
        user_rating = self.ratings_matrix.loc[self.userId, :]
        seen_movies = user_rating[user_rating>0].index.tolist()
        movie_list = self.ratings_matrix.columns.tolist()
        unseen_movies = [movie for movie in movie_list if movie not in seen_movies]

        return unseen_movies

    def get_recommend_movie(self):
        unseen_movies = self.get_unseen_movies()
        predict_matrix = pd.DataFrame(data=self.predict, index=self.ratings_matrix.index, columns=self.ratings_matrix.columns)
        recommend_movies = predict_matrix.loc[self.userId, unseen_movies].sort_values(ascending=False)[:self.top_n]
        recomm_movies = pd.DataFrame(data=recommend_movies.values, index=recommend_movies.index, columns=['pred_score'])

        return recomm_movies

    def run(self):
        self.matrix_factorization()
        predicted_df = pd.DataFrame(data=self.predict, index=self.ratings_matrix.index, columns=self.ratings_matrix.columns)
        return self.get_recommend_movie()


if __name__ == '__main__':
    movies = pd.read_csv('~/ml-latest-small/movies.csv')
    ratings = pd.read_csv('~/ml-latest-small/ratings.csv')
    matrixfactorization = MatrixFactorizeRecommendation(movies=movies, ratings=ratings, userId=9, top_n=7)
    print(matrixfactorization.run())
