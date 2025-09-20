import os
from dotenv import load_dotenv
from datetime import datetime
from logging_custom.logger import Logger
import pandas as pd
import streamlit as st
import numpy as np
from dataframe_manager.manage_dataframe import DataFrameManager

load_dotenv()

USERS_DATA = os.getenv("USERS_DATA")

class UserHelper:
    def __init__(self):
        self.logger = Logger("UserHelper").get_logger()
        self.users_df = self.load_users_data_cached(USERS_DATA)
        # Set columns and index once after loading
        self.users_df.columns = ['userId', 'avg_rating', 'avg_hour']
        self.users_df.set_index('userId', inplace=True)
        self.logger.info(f"User dataframe loaded from {USERS_DATA}")

    @staticmethod
    @st.cache_data
    def load_users_data_cached(file_path):
        if file_path and os.path.exists(file_path):
            return DataFrameManager(file_path).load_dataframe()
        else:
            raise FileNotFoundError(f"Users data path {file_path} does not exist.")
    
    def get_user_vector(self, userId):
        self.logger.info(f"Generating user genre preference vector for userId {userId}.")
        if userId not in self.users_df.index:
            self.logger.error(f"User ID {userId} not found in users data.")
            raise ValueError(f"User ID {userId} not found in users data.")
        user_vector = np.array(self.users_df.loc[userId]).reshape(1, -1)
        return user_vector
    
    def update_user_data_from_ratings(self, ratings_df):
        self.logger.info(f"Updating users data from ratings data")
        ratings_df['hour'] = ratings_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x).hour)
        # users data
        users_df_new = ratings_df.groupby('userId').agg(
            avg_rating=('rating', 'mean'),
            avg_hour=('hour', 'mean')
        ).reset_index()

        if len(users_df_new) > len(self.users_df):
            users_df_new.to_csv(USERS_DATA, index=False)
            self.users_df = users_df_new.set_index('userId')
            self.logger.info(f"Users data updated and saved to {USERS_DATA}")
            return True
        else:
            self.logger.info(f"Users data length mismatch. Previous users data len: {len(self.users_df)}, New Users data len: {len(users_df_new)}")
            raise ValueError(f"Users data length mismatch. Previous users data len: {len(self.users_df)}, New Users data len: {len(users_df_new)}")


