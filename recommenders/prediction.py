import streamlit as st
import numpy as np
from collections import defaultdict
from logging_custom.logger import Logger
from recommenders.cmf_recommender import CMFRecommender
from recommenders.xgboost_recommender import XGBoostRecommender

class Prediction:
    def __init__(self, user_helper, movie_helper, links_helper, ratings_helper):
        self.logger = Logger("Prediction").get_logger()
        # Initialize helper functions
        self.user_helper = user_helper
        self.movie_helper = movie_helper
        self.links_helper = links_helper
        self.ratings_helper = ratings_helper

        self.logger.info("Prediction class initialized.")
    
    def predict_cmf_user_movie_score(self, userId, movieId):
        self.logger.info(f"Predicting score for userId {userId} and movieId {movieId} using CMF Recommender.")
        cmf_recommender = CMFRecommender()
        score = cmf_recommender.predict([userId], [movieId])[0][1]
        return score
    
    def predict_cmf_topN(self, userId, N=10):
        self.logger.info(f"Fetching top {N} recommendations for userId {userId} using CMF Recommender.")
        cmf_recommender = CMFRecommender()
        seen_movies = self.ratings_helper.get_user_movie_seen(userId)
        preds = cmf_recommender.topN_predict(userId, seen_movies, N)
        return preds
    
    def cmf_simlar_movies(self, movieId):
        self.logger.info(f"Similar movies to movie {movieId} using CMF Recommender item matrix.")
        cmf_recommender = CMFRecommender()
        similar_movies = cmf_recommender.item_item_sim_optimized(movieId)
        return similar_movies
    
    def cmf_simlar_users(self, userId: int):
        self.logger.info(f"Similar users to user {userId} using CMF Recommender user matrix.")
        cmf_recommender = CMFRecommender()
        similar_users = cmf_recommender.user_user_sim(userId)
        return similar_users
    
    def _get_aggregated_movie_scores(self, candidate_movies: dict):
        if candidate_movies:
            scores = defaultdict(list)
            for movies in candidate_movies.values():
                for movie_id, w_scores in movies:
                    scores[movie_id].append(w_scores)
            # sum the scores for multiple occurence
            aggregated_movie_scores = {movie_id: sum(w_scores) for movie_id, w_scores in scores.items()}
            return aggregated_movie_scores

    @st.cache_data
    def cmf_similar_users_optimized(_self, userId: int):
        _self.logger.info(f"Attempting user similarity search for {userId}")
        # # initialize cmf class
        # cmf_recommender = CMFRecommender()
        # find similar users
        top_similar_users = CMFRecommender().user_user_sim_optimized(userId)
        movies_seen_by_user = _self.ratings_helper.get_user_movie_seen(userId) # movies seen by current user
        _self.logger.info(f"number of movies seen by user: {len(movies_seen_by_user)}")
        # find top movies seen by similar users, user hasnt watched
        candidate_movies = {}
        for user_id, sim in top_similar_users:
            if user_id != userId:
                movies_seen_by_similar_user = _self.ratings_helper.get_user_movie_seen(user_id) # movies seen by similar users
                _self.logger.info(f"number of movies seen by similar user: {len(movies_seen_by_similar_user)}")
                # get rid of movies, current user already seen
                recommendable_movies = list(set(movies_seen_by_similar_user) - set(movies_seen_by_user))
                _self.logger.info(f"recommendable movies from user: {user_id} : {recommendable_movies}")
                recommendable_movies = [(movie_id, _self.ratings_helper.get_user_movie_rating(user_id, movie_id)) for movie_id in recommendable_movies]
                # add movie, ratings*similarity in candidate movie dict
                candidate_movies[user_id] = [(movie_id, rating * sim) for movie_id, rating in recommendable_movies]
        # aggregrate movies and scores (2 or more users may like the same movie)
        aggregated_movie_scores = _self._get_aggregated_movie_scores(candidate_movies)
        # return the sorted movie ids based on scores
        return sorted(aggregated_movie_scores.items(), key=lambda x: x[1], reverse=True)
                
                

    
    def predict_xgboost(self, userId, N=10):
        self.logger.info(f"Fetching top {N} recommendations for userId {userId} using XGBoost Recommender.")
        # load movie and user vectors
        movie_ids, movie_vectors = self.movie_helper.get_random_movie_vectors(n=100)
        user_vector = self.user_helper.get_user_vector(userId)
        seen_movies = self.ratings_helper.get_user_movie_seen(userId)
        # load model
        xgb_recommender = XGBoostRecommender()
        # scale user vector
        user_vector = xgb_recommender.preprocess(user_vector)
        # create user vector repeated for each movie
        user_vector = user_vector.repeat(len(movie_vectors), axis=0)
        # create input for prediction by combining user and movie vectors
        X_input = np.hstack((user_vector, movie_vectors))
        # get predictions
        preds = xgb_recommender.predict(X_input, movie_ids, seen_movies)
        
        return preds[:N]
    
    

        
