import numpy as np
import pandas as pd
from datetime import datetime
from logging_custom.logger import Logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from recommenders.xgboost_recommender import XGBoostRecommender

class ColdStartRecommender:
    def __init__(self, movie_helper):
        self.logger = Logger("cold_start").get_logger()
        self.movie_helper = movie_helper
        # Pivoted genre DataFrame: index=movieId, columns=genres, values=0/1
        self.movie_genre_pivot = self.movie_helper.pivot_genres()

    def recommend(self, new_user_vector, liked_movies, top_n=20, threshold=0.8):
        """
        Recommend top_n movies most similar to new_user_vector, excluding liked_movies.
        Args:
            new_user_vector: np.array of shape (num_genres,)
            liked_movies: set or list of movieIds to exclude
            top_n: number of movies to recommend
            threshold: minimum cosine similarity
        Returns:
            List of movieIds
        """
        top_movies = []
        # Randomize order for diversity
        for mid in np.random.choice(self.movie_genre_pivot.index, size=len(self.movie_genre_pivot.index), replace=False):
            if mid in liked_movies:
                continue
            movie_vector = np.array(self.movie_genre_pivot.loc[mid]).reshape(1, -1)
            similarity = cosine_similarity(new_user_vector.reshape(1, -1), movie_vector)[0][0]
            if similarity >= threshold:
                top_movies.append(mid)
                if len(top_movies) == top_n:
                    self.logger.info(f"found top {top_n} movies.")
                    break
        # also return a user vector to check user preferences
        new_user_vector = pd.DataFrame(new_user_vector.reshape(1, -1), columns=self.movie_genre_pivot.columns.tolist())
        new_user_vector = new_user_vector.T
        new_user_vector.columns = ['Genre score']

        return top_movies, new_user_vector.round(2)

    def get_movie_vector(self, movie_id):
        """Get the genre vector for a given movie_id."""
        return np.array(self.movie_genre_pivot.loc[movie_id])
    
    def recommend_from_liked(self, liked_movie_ids, top_n=20, threshold=0.8):
        """
        Build new user vector from liked_movie_ids and return top recommendations.
        Args:
            liked_movie_ids: list or set of movieIds liked by the user
            top_n: number of movies to recommend
            threshold: minimum cosine similarity
        Returns:
            List of recommended movieIds
        """
        if not liked_movie_ids:
            raise ValueError("No liked movies provided.")
        self.logger.info(f"Similarity search using genres. num results requested: {top_n}, similarity threshold: {threshold}")
        # Build new user vector by averaging genre vectors of liked movies
        liked_vectors = [self.get_movie_vector(mid) for mid in liked_movie_ids if mid in self.movie_genre_pivot.index]
        if not liked_vectors:
            raise ValueError("None of the liked movies found in genre pivot table.")
        new_user_vector = np.mean(liked_vectors, axis=0)
        self.logger.info(f"New user vector:::: {new_user_vector}")
        # Use recommend method to get top movies
        return self.recommend(new_user_vector, set(liked_movie_ids), top_n=top_n, threshold=threshold)
    
    def xgb_cold_start(self, new_user_info: pd.DataFrame):
        self.logger.info(f"Predicting movies for new user using XGB Recommender")
        seen_movies = new_user_info['movieId'].values.tolist()
        new_user_info['hour'] = new_user_info['timestamp'].apply(lambda x: datetime.fromtimestamp(x).hour)
        # convert user infos
        avg_rating = new_user_info.groupby('userId')['rating'].mean()
        avg_hour = new_user_info.groupby('userId')['hour'].mean()
        # get random movies
        movie_ids, movie_vectors = self.movie_helper.get_random_movie_vectors(n=100)
        # initialize xgb recommender
        xgb = XGBoostRecommender()
        # preprocess user data
        user_vector = xgb.preprocess(pd.DataFrame({'avg_rating': avg_rating, 'avg_hour': avg_hour}))
        # create user vector repeated for each movie
        user_vector = user_vector.repeat(len(movie_vectors), axis=0)
        # create input for prediction by combining user and movie vectors
        X_input = np.hstack((user_vector, movie_vectors))
        # get predictions
        preds = xgb.predict(X_input, movie_ids, seen_movies)

        return preds


    
    def get_user_preference_vector(self, liked_movie_ids, ratings):
        if not liked_movie_ids or not ratings:
            self.logger.error(f"Movie Ids or Ratings are missing. {liked_movie_ids}, {ratings}")
            raise ValueError(f"Movie Ids or Ratings are missing. {liked_movie_ids}, {ratings}")
        # get movie genre vectors
        liked_vectors = [self.get_movie_vector(mid) for mid in liked_movie_ids if mid in self.movie_genre_pivot.index]
        self.logger.info(f"liked vector: {liked_vectors}")
        # multiply movie genre vectors with ratings
        new_user_vector = [rating*movie_vector for rating, movie_vector in zip(ratings, liked_vectors)]
        new_user_vector = np.mean(new_user_vector, axis=0)
        self.logger.info(f"New user vector: {new_user_vector}")
        new_user_vector = pd.DataFrame(new_user_vector.reshape(1, -1), columns=self.movie_genre_pivot.columns.tolist())
        new_user_vector = new_user_vector.T
        new_user_vector.columns = ['Genre score']
        new_user_vector['Genre score'] = MinMaxScaler().fit_transform(new_user_vector)
        return new_user_vector

    
    
