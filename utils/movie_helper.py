import os
from dotenv import load_dotenv
from logging_custom.logger import Logger
import streamlit as st
import numpy as np

from dataframe_manager.manage_dataframe import DataFrameManager

load_dotenv()

MOVIES_DATA = os.getenv("MOVIES_DATA")

class MovieHelper:
    def __init__(self):
        self.logger = Logger("MovieHelper").get_logger()
        self.movies_df = self.load_movies_data_cached(MOVIES_DATA)
        self.logger.info(f"Movies dataframe loaded from {MOVIES_DATA}")

    @staticmethod
    @st.cache_data
    def load_movies_data_cached(file_path):
        if file_path and os.path.exists(file_path):
            return DataFrameManager(file_path).load_dataframe()
        else:
            raise FileNotFoundError(f"Movies data path {file_path} does not exist.")
    
    def explode_genres(self):
        self.logger.info("Exploding genres into separate rows.")
        # self.movies_df['genres'] = self.movies_df['genres'].apply(lambda x: "none" if x == '(no genres listed)' else x)
        # self.movies_df['genres'] = self.movies_df['genres'].fillna('none')
        genre_split_df = self.movies_df.copy()
        genre_split_df['genres'] = self.movies_df['genres'].str.split('|')
        self.logger.info(f"Splitted genres sample: {genre_split_df.head()}")
        exploded_df = genre_split_df.explode('genres').reset_index(drop=True)
        self.logger.info(f"Splitted genres sample: {exploded_df.head()}")
        return exploded_df
    
    def pivot_genres(self):
        self.logger.info("Pivoting genres into binary vector format.")
        exploded_df = self.explode_genres()
        pivoted_df = exploded_df.pivot(index='movieId', columns='genres', values='title')
        self.logger.info(f"Pivoted df shape: {pivoted_df.shape}, Columns: {pivoted_df.columns}")
        pivoted_df = pivoted_df.notnull().astype(int)
        pivoted_df = pivoted_df.drop(columns=['(no genres listed)'], errors='ignore')
        self.logger.info(f"Pivoted df shape: {pivoted_df.shape}, Columns: {pivoted_df.columns}")
        return pivoted_df
    
    def get_movie_vector(self, movieId):
        self.logger.info("Generating movie genre vector.")
        genre_vector_df = self.pivot_genres()
        movie_vector = np.array(genre_vector_df.loc[movieId]).reshape(1, -1)
        return movie_vector
    
    def get_choice_movie_vectors(self, movieIds):
        self.logger.info("Generating choice movie genre vectors.")
        choice_movie_vectors = np.array([self.get_movie_vector(mid).flatten() for mid in movieIds])
        return choice_movie_vectors
    
    def get_random_movies(self, n=10, rand_state=42):
        self.logger.info(f"Selecting {n} random choice movies.")
        choice_movies = self.movies_df.sample(n=n, random_state=rand_state)[['movieId', 'title', 'genres']]
        return choice_movies
    
    def get_random_movie_vectors(self, n=100):
        self.logger.info(f"Generating genre vectors for {n} random movies.")
        pivoted_df = self.pivot_genres()
        self.logger.info(f"Pivoted df columns (genres): {pivoted_df.columns}")
        random_movies = pivoted_df.sample(n=n)
        self.logger.info(f"Random movies shape: {random_movies.shape}")
        movieIds = random_movies.index.tolist()
        random_movie_vectors = random_movies.values
        return movieIds, random_movie_vectors
        


    
