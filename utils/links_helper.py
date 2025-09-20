import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from logging_custom.logger import Logger

from dataframe_manager.manage_dataframe import DataFrameManager

load_dotenv()
LINKS_DATA = os.getenv("LINKS_DATA")

class LinksHelper:
    def __init__(self):
        self.logger = Logger("LinksHelper").get_logger()
        self.links_df = self.load_links_data_cached(LINKS_DATA)
        self.logger.info(f"Links dataframe loaded from {LINKS_DATA}")

    @staticmethod
    @st.cache_data
    def load_links_data_cached(file_path):
        if file_path and os.path.exists(file_path):
            return DataFrameManager(file_path).load_dataframe()
        else:
            raise FileNotFoundError(f"Links data path {file_path} does not exist.")
    
    def get_imdb_id(self, movieId):
        # self.logger.info(f"Fetching IMDb ID for movieId {movieId}")
        imdb_id = str(self.links_df[self.links_df['movieId'] == movieId]['imdbId'].values[0])
        # condition to match id
        zero_prefix = 0
        if len(imdb_id) < 7:
            zero_prefix = 7 - len(imdb_id)
        imdb_id = 'tt' + '0' * zero_prefix + imdb_id
        
        return imdb_id if len(imdb_id) > 0 else None
    
    def get_tmdb_id(self, movieId):
        self.logger.info(f"Fetching TMDb ID for movieId {movieId}")
        tmdb_id = self.links_df[self.links_df['movieId'] == movieId]['tmdbId'].values
        return tmdb_id
    
    def search_img_data(self, movieId):
        self.logger.info(f"Searching image data for movieId {movieId}")
        base64_str = self.links_df[self.links_df['movieId'] == movieId]['base64_image'].values[0]
        if pd.notna(base64_str) and base64_str != '':
            return base64_str
        else:
            return None
    
    def update_links_img_data(self, movieId, base64_str):
        self.logger.info("Updating links data.")
        self.links_df.loc[self.links_df['movieId'] == movieId, 'base64_image'] = base64_str
        self.links_df.to_csv(LINKS_DATA, index=False)
        self.logger.info(f"Links data updated and saved to {LINKS_DATA}.")