import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import streamlit as st
from logging_custom.logger import Logger

load_dotenv()

RATINGS_DATA = os.getenv("RATINGS_DATA")

class RatingsHelper:
    def __init__(self):
        self.logger = Logger("RatingsHelper").get_logger()
        self.ratings_df = self.load_ratings_data_cached(RATINGS_DATA)
        self.logger.info(f"Ratings data loaded from {RATINGS_DATA}")

    @staticmethod
    @st.cache_data
    def load_ratings_data_cached(file_path):
        if file_path and os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"Ratings data path {file_path} does not exist.")
    
    def update_ratings(self, userId, movieId, rating, timestamp):
        self.logger.info(f"Updating rating for movie {movieId} by user {userId}: {rating}")
        # Check if the user-movie pair already exists
        mask = (self.ratings_df['userId'] == userId) & (self.ratings_df['movieId'] == movieId)
        if self.ratings_df[mask].empty:
            # Insert new rating
            new_row = {'userId': [userId], 'movieId': [movieId], 'rating': [rating], 'timestamp': [timestamp]}
            self.ratings_df = pd.concat([self.ratings_df, pd.DataFrame(new_row)], ignore_index=True)
            self.logger.info("Inserted new rating.")
        else:
            # Update existing rating
            self.ratings_df.loc[mask, 'rating'] = rating
            self.ratings_df.loc[mask, 'timestamp'] = int(np.datetime64('now').astype('int64') // 1e9)
            self.logger.info("Updated existing rating.")
        return True
    
    def get_user_movie_seen(self, userId):
        self.logger.info(f"Fetching movies seen by userId {userId}")
        seen_movies = self.ratings_df[self.ratings_df['userId'] == userId]['movieId'].values.tolist()
        return seen_movies
    
    def get_user_movie_rating(self, userId, movieId):
        rating = self.ratings_df.loc[(self.ratings_df['userId'] == userId) & (self.ratings_df['movieId'] == movieId)]['rating'].values[0]
        # self.logger.info(f"Rating by user {userId} to movie {movieId}: {rating}")
        return rating
    
    def update_ratings_from_new_ratings(self, users_helper):
        new_ratings_path = os.path.join(os.path.dirname(RATINGS_DATA), "new_ratings.csv")
        if os.path.exists(new_ratings_path):
            new_ratings_df = pd.read_csv(new_ratings_path)
            self.logger.info(f"Ratings shape before concat: {self.ratings_df.shape}")
            combined_df = pd.concat([self.ratings_df, new_ratings_df], ignore_index=True)
            self.logger.info(f"Combined df shape after concat: {combined_df.shape}")
            if combined_df.shape[0] == self.ratings_df.shape[0] + new_ratings_df.shape[0]:
                # Keep only the last occurrence for each userId-movieId pair
                combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['userId', 'movieId'], keep='last')
                self.ratings_df = combined_df
                self.logger.info(f"Final Ratings shape after concat and dropping duplicates: {self.ratings_df.shape}")
                self.ratings_df.to_csv(RATINGS_DATA, index=False)
                self.logger.info(f"Ratings data updated and saved to {RATINGS_DATA}")
                # also update the users data
                users_helper.update_user_data_from_ratings(combined_df)
                return True
            else:
                self.logger.error("Before and after ratings shape are not correct.")
                raise ValueError("Mistmatch in num ratings and combined dataframe. Please try again.")
        else:
            self.logger.warning(f"new_ratings.csv not found at {new_ratings_path}")
    