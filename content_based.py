import pandas as pd
import warnings
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')


class ContentBasedRecommendation:
    def __init__(self, movie_data, top_n):
        """ The algorithm is based on item-based systems. It will return top n recommended items for input items.
        :param movie_data: tmdb_5000_movies
        :param top_n: Get n recommended items
        """
        self.movie_data = movie_data
        self.top_n = top_n

    def data_preprocessing(self):
        self.movie_data['genres'] = self.movie_data['genres'].apply(literal_eval)
        self.movie_data['keywords'] = self.movie_data['keywords'].apply(literal_eval)
        self.movie_data['genres'] = self.movie_data['genres'].apply(lambda x: [y['name'] for y in x])
        self.movie_data['keywords'] = self.movie_data['keywords'].apply(lambda x: [y['name'] for y in x])
        return

    def vectorize_genres(self):
        """ Vectorize genres using CountVectorizer.
        :return: genre similarity
        """
        self.movie_data['genres_str'] = self.movie_data['genres'].apply(lambda x: ' '.join(x))
        genre_vectorizer = CountVectorizer()
        gerne_matrix = genre_vectorizer.fit_transform(self.movie_data['genres_str'])
        genre_sim = cosine_similarity(gerne_matrix, gerne_matrix)
        sorted_genre_sim = genre_sim.argsort()[:, ::-1]
        return sorted_genre_sim

    def get_sim_movies(self, sorted_idx, title):
        """ Split recommendation system to two steps to improve accuracy. First, get top n*2 items using only
        genre similarity. Second, rank top n*2 items by weighted_vote.
        :param sorted_idx:
        :param title:
        :return: top n recommended items
        """
        title_movie = self.movie_data[self.movie_data['title'] == title]
        title_idx = title_movie.index.values
        similar_indices = sorted_idx[title_idx, :(self.top_n*2)]
        similar_indices = similar_indices.reshape(-1)
        similar_indices = similar_indices[similar_indices != title_idx]
        return self.movie_data.iloc[similar_indices].sort_values('weighted_vote', ascending=False)[:self.top_n]

    def cal_weighted_vote_average(self, info):
        m = self.movie_data['vote_count'].quantile(0.6)
        C = self.movie_data['vote_average'].mean()
        v = info['vote_count']
        R = info['vote_average']
        return (v/(v+m) * R + (m/(m+v)) * C)

    def run(self):
        self.movie_data = self.movie_data[['genres', 'id', 'keywords', 'popularity', 'title', 'vote_average', 'vote_count']]
        self.data_preprocessing()
        self.movie_data['weighted_vote'] = self.movie_data.apply(lambda x: self.cal_weighted_vote_average(x), axis=1)
        genre_similarity = self.vectorize_genres()
        recommend_movies = self.get_sim_movies(genre_similarity, 'The Godfather')
        return recommend_movies[['title', 'vote_average', 'weighted_vote']]


if __name__ == '__main__':
    data = pd.read_csv('~/archive/tmdb_5000_movies.csv')
    recommendation_system = ContentBasedRecommendation(movie_data=data, top_n=7)
    recommend_res = recommendation_system.run()
    print(recommend_res)
