import os
from dotenv import load_dotenv
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from logging_custom.logger import Logger
import streamlit as st

load_dotenv()

MODEL_PATH = os.getenv("CMF_MODEL_PATH")
CMF_USER_KNN = os.getenv("CMF_USER_KNN")
CMF_ITEM_KNN = os.getenv("CMF_ITEM_KNN")

class CMFRecommender:
    def __init__(self):
        self.logger = Logger("CMFRecommender").get_logger()
        self.model = self.load_model()
        self.similarity_threshold = 0.5
        self.logger.info("CMF Recommender initialized.")

    @staticmethod
    @st.cache_data
    def load_model():
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            # self.logger.info(f"Loading CMF model from {MODEL_PATH}")
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            # self.logger.error(f"Model path {MODEL_PATH} does not exist.")
            raise FileNotFoundError(f"Model path {MODEL_PATH} does not exist.")
    
    def predict(self, userId, movieId):
        self.logger.info("Making predictions.")
        try:
            preds = self.model.predict(userId, movieId)
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            raise e
        return preds
    
    def topN_predict(self, userId, seen_movies, N=10):
        self.logger.info(f"Fetching top {N} recommendations for userId {userId}.")
        if self.model is None:
            self.logger.error("Model is not loaded. Cannot generate predictions.")
            raise ValueError("Model is not loaded. Cannot generate predictions.")

        # --- Enhanced Debugging Starts Here ---
        self.logger.info(f"Attempting to get topN for userId: {userId}")
        self.logger.info(f"Number of recommendations requested (N): {N}")
        self.logger.info(f"Number of movies to exclude: {len(seen_movies)}")
        
        # Check if the userId exists in the model's user mapping
        try:
            # The 'user_mapping' or similar attribute may vary by library.
            # This is a common pattern for CMFrec.
            if userId not in self.model.user_mapping_:
                self.logger.warning(f"UserId {userId} not found in model's user mapping.")
                raise ValueError(f"UserId {userId} not found in model's user mapping.")
        except AttributeError:
            self.logger.warning("Could not check user mapping. Model object might not have this attribute.")

        try:
            preds = self.model.topN(userId, N, exclude=seen_movies, output_score=True)
            self.logger.info("TopN prediction completed successfully.")
            preds = list(zip(preds[0], preds[1]))
            preds = sorted(preds, key=lambda x: x[1], reverse=True)
            return preds

        except Exception as e:
            # Catch the exception and log the full traceback for a complete diagnosis
            self.logger.error(f"Error in topN prediction: {e}", exc_info=True)
            raise e
        # --- Enhanced Debugging Ends Here ---
    def user_user_sim(self, user_to_check: int):
        self.logger.info(f"Checking for similar users in the database for {user_to_check}")
        try:
            if user_to_check:
                user_to_check_index = np.squeeze(np.where(self.model.user_mapping_ == user_to_check))
                user_to_check_vector = self.model.A_[user_to_check_index]
                sim_user_dict = {}
                
                for i, u_vec in enumerate(self.model.A_):
                    
                    u_id = int(self.model.user_mapping_[i])
                    if u_id != user_to_check:
                        similarity = cosine_similarity(user_to_check_vector.reshape(1, -1), u_vec.reshape(1, -1))[0][0]
                        if similarity > self.similarity_threshold:
                            sim_user_dict[u_id] = similarity
                
                sim_user_dict = dict(sorted(sim_user_dict.items(), key=lambda x: x[1], reverse=True))
                
                return sim_user_dict
        except Exception as e:
            self.logger.error(f"Error in users similarity search: {str(e)}")
            raise e
    
    def item_item_sim(self, item_to_check: int):
        self.logger.info(f"Checking for similar movies in the database for {item_to_check}")
        try:
            if item_to_check:
                item_to_check_index = np.squeeze(np.where(self.model.item_mapping_ == item_to_check))
                item_to_check_vector = self.model.B_[item_to_check_index]
                sim_item_dict = {}
                
                for i, i_vec in enumerate(self.model.B_):
                    
                    i_id = int(self.model.item_mapping_[i])
                    if i_id != item_to_check:
                        similarity = cosine_similarity(item_to_check_vector.reshape(1, -1), i_vec.reshape(1, -1))[0][0]
                        if similarity > self.similarity_threshold:
                            sim_item_dict[i_id] = similarity
                
                sim_item_dict = dict(sorted(sim_item_dict.items(), key=lambda x: x[1], reverse=True))
                self.logger.info(f"Number of similar movies found: {len(sim_item_dict)}")
                return sim_item_dict
        except Exception as e:
            self.logger.error(f"Error in movies similarity search: {str(e)}")
            raise e
        
    def users_kneighbors(self, user_idx):
        self.logger.info(f"get Nearest Neighbors for user index {user_idx}")
        if user_idx:
            try:
                with open(CMF_USER_KNN, 'rb') as f:
                    nn_user = pickle.load(f)
                return nn_user.kneighbors(self.model.A_[user_idx].reshape(1, -1))
            except Exception as e:
                self.logger.error(f"Error KNeighbors prediction: {e}")
                raise e
        else:
            raise ValueError("No user indices to parse")
    
    def item_kneighbors(self, item_idx):
        self.logger.info(f"get Nearest Neighbors for item index {item_idx}")
        if item_idx:
            try:
                with open(CMF_ITEM_KNN, 'rb') as f:
                    nn_item = pickle.load(f)
                return nn_item.kneighbors(self.model.B_[item_idx].reshape(1, -1))
            except Exception as e:
                self.logger.error(f"Error KNeighbors prediction: {e}")
                raise e
        else:
            raise ValueError("Failed to parse item indices")

    def user_user_sim_optimized(self, user_to_check: int):

        self.logger.info(f"Finding similar users to user {user_to_check}")
        # user index from model user mapping
        if user_to_check not in self.model.user_mapping_:
            self.logger.error(f"No user found in user mapping for user {user_to_check}")
            raise ValueError(f"User {user_to_check} not exist in model user mapping")
        user_to_check_index = np.where(self.model.user_mapping_ == user_to_check)[0][0]
        # get Nearest Neighbors
        distances, users_idx = self.users_kneighbors(user_to_check_index)
        # get user ids
        users_id = self.model.user_mapping_[users_idx[0]]
        top_similar_users = list(zip(users_id, distances[0]))
        # sort users idx by distances -> [(usr_id_1, dist_1), (usr_id_2, dist_2), ....]
        top_similar_users = sorted(top_similar_users, key=lambda x: x[1], reverse=True)
        self.logger.info(f"top similar users predicted {top_similar_users}")
        return top_similar_users[:10]
    
    def item_item_sim_optimized(self, item_to_check: int):
        self.logger.info(f"Finding similar item to item {item_to_check}")
        # item index from model user mapping
        if item_to_check not in self.model.item_mapping_:
            self.logger.error(f"No item found in item mapping for item {item_to_check}")
            raise ValueError(f"Item {item_to_check} does not exist in model item mapping")
        # get item index from item mapping
        item_to_check_index = np.where(self.model.item_mapping_ == item_to_check)[0][0]
        # get nearest neighbors to item
        distances, items_idx = self.item_kneighbors(item_to_check_index)
        # get item ids
        items_id = self.model.item_mapping_[items_idx[0]]
        top_similar_items = list(zip(items_id, distances[0]))
        # sort by distances
        top_similar_items = sorted(top_similar_items, key=lambda x: x[1], reverse=True)
        self.logger.info(f"top similar items predicted {top_similar_items}")
        return top_similar_items




        
