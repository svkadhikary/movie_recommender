import os
from dotenv import load_dotenv
import streamlit as st
from utils.ratings_helper import RatingsHelper
from utils.user_helper import UserHelper
from dataframe_manager.manage_dataframe import DataFrameManager
from model_trainer.cmf_trainer import CMFTrainer
from logging_custom.logger import Logger

load_dotenv()
# Initialize logger
logger = Logger("app_logger").get_logger()
# Set page configuration
st.set_page_config(page_title="Movie Recommender App", layout="wide")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)
    st.title(":blue[Movie Recommender App] :clapper:")
    st.write("Welcome to the Movie Recommender App! Please log in to continue.")

# login functionality
def login(userId, password):
    logger.info(f"Attempting login for userId: {userId}")
    user_path = os.getenv("USERS_DATA")
    logger.debug(f"User data path from environment variable: {user_path}")
    if user_path is None:
        logger.error("Users data environment variable not set or path does not exist.")
        return False
    df_mgr = DataFrameManager(str(user_path))
    df = df_mgr.load_dataframe()
    user = df[df['userId'] == userId]
    if not user.empty:
        logger.info(f"User {userId} found in database.")
        return True
    return False

# Login form
with col2:
    with st.form("login_form"):
        userId = st.text_input("userId")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if login(int(userId), password):
                st.success("Login successful!")
                st.session_state['logged_in'] = True
                st.session_state['userId'] = userId
                st.text(f"You are now logged in. User: {userId}")
                # Redirect to dashboard
                st.switch_page("pages/dashboard.py")
            else:
                
                st.error("Invalid userId or password.")
    cols = st.columns(2)
    with cols[1]:
        update_model = st.button("Update Model", key="update_model", help="Update recommender models. *Please update ratings first*")
    with cols[0]:
        update_ratings = st.button("Update Ratings", key="update_ratings", help="For updating the ratings and user registrations.")

# update model
if update_model:
    cmf = CMFTrainer()
    try:
        with st.spinner("CMF training in progress...", show_time=True):
            if cmf.search_best_param():
                st.success("Best Model trained and saved. Close the app and reopen to initialize changes.")
    except Exception as e:
        logger.error(f"Error occured: {e}")
        st.error(f"Error occured while training: {e}")
    # st.warning("Currently models are not being updated due to certain unforseen circumstances. Please come back later...")
    
    
# update ratings
if update_ratings:
    ratings_helper = RatingsHelper()
    users_helper = UserHelper()
    if ratings_helper.update_ratings_from_new_ratings(users_helper):
        st.success("Ratings data updated. Train models to get recommendations")
    else:
        st.error("Something didn't work right. Try again later...")
    
