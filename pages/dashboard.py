import os
from dotenv import load_dotenv
import numpy as np
from logging_custom.logger import Logger
from utils import movie_helper, user_helper, links_helper, ratings_helper
from utils.get_poster import get_image
from recommenders.prediction import Prediction
from recommenders.cold_start import ColdStartRecommender

import streamlit as st

load_dotenv()
# Redirect to home if not logged in
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.session_state['logged_in'] = False
    st.switch_page("app.py")
# get user id
userId = st.session_state.get('userId', None)
# Initialize logger
logger = Logger("dashboard_page").get_logger()
# Set page configuration
st.set_page_config(page_title="Movie Recommender App", layout="wide")

movie_helper = movie_helper.MovieHelper()
user_helper = user_helper.UserHelper()
links_helper = links_helper.LinksHelper()
ratings_helper = ratings_helper.RatingsHelper()

prediction = Prediction(user_helper, movie_helper, links_helper, ratings_helper)
# Function to get recommendations based on model choice
def predict_recommendations(userId, model_option, N=10):
    if userId is None:
        logger.error("User ID is None. Cannot fetch recommendations.")
        raise ValueError("User ID is None. Cannot fetch recommendations.")
        
    logger.info(f"Predicting recommendations for userId {userId} using {model_option}.")
    # get recommendations
    if model_option == "XGBoost Recommender":
        preds = prediction.predict_xgboost(int(userId), N=N)
    elif model_option == "CMF Recommender":
        preds = prediction.predict_cmf_topN(int(userId), N=N)
    elif model_option == "User Similarity Based":
        preds = prediction.cmf_similar_users_optimized(int(userId))[:N]
    else:
        logger.error(f"Model option {model_option} not recognized.")
        raise ValueError(f"Model option {model_option} not recognized.")
    # get imdb ids
    logger.info("Fetching IMDb IDs for recommended movies.")
    preds = [(movieId, score, links_helper.get_imdb_id(movieId)) for movieId, score in preds]
    return preds
# Search movies
def search_movies(search_query):
    return movie_helper.movies_df[movie_helper.movies_df['title'].str.contains(search_query, case=False, na=False)]

col1, col2 = st.columns([4, 1])
with col2:
    # Logout button
    st.button("Logout", on_click=lambda: st.session_state.update({'logged_in': False, 'userId': None}))
    # user profile
    st.write("User Profile")
    # get ratings data for user
    user_profile = ratings_helper.ratings_df[ratings_helper.ratings_df['userId'] == int(userId)][['movieId', 'rating']]
    # seperate movie and ratings
    likes_ids = user_profile['movieId'].values.tolist()
    ratings = user_profile['rating'].values.tolist()
    # get user preference df
    user_vec_df = ColdStartRecommender(movie_helper).get_user_preference_vector(likes_ids, ratings)
    # round and sort values
    user_vec_df = user_vec_df.round(2)
    user_vec_df = user_vec_df.sort_values(by='Genre score', ascending=False)
    # display user preference dataframe
    st.dataframe(user_vec_df.style.background_gradient(cmap='summer', axis=None), height=600)
    # go to user profile
    if st.button("Go to user profile", key='user_profile'):
        st.switch_page("pages/user_profile.py")

with st.sidebar:
    search = st.text_input("Search movies", key='search_movies', placeholder='Avengers...')
    if search:
        results = search_movies(search)
        for _, row in results.iterrows():
            if st.button(row['title'], key=f"search_{row['movieId']}"):
                st.session_state['movieId'] = int(row['movieId'])
                st.session_state['imdb'] = links_helper.get_imdb_id(row['movieId'])
                st.switch_page('pages/movie_page.py')
    st.title("Settings :gear:")
    model_option = st.selectbox("Choose Recommendation Model", ("CMF Recommender", "XGBoost Recommender", "User Similarity Based"))

with col1:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)
    st.title(f":grey[Hello User {userId}] :movie_camera:")
    st.write("Welcome to the Movie Recommender App! :star:")
    st.write("Chhose your desired model from the sidebar. :rocket:")
    st.write(f"Here are some recommendations based on your profile using **{model_option}**. :sparkles:")
    # display recommendations
    try:
        recommendations = predict_recommendations(userId, model_option, N=8)
        st.subheader("Top Movie Recommendations for You:")

        for i, (movieId, score, imdb) in enumerate(recommendations):
            if i % 4 == 0:
                cols = st.columns(4)
            
            with cols[i % 4]:
                movie_title = movie_helper.movies_df[movie_helper.movies_df['movieId'] == movieId]['title'].values[0]
                # st.write(f"**{movie_title}** (Movie ID: {movieId}) - Predicted Score: {score:.2f}")
                st.image(get_image(imdb, movieId, links_helper), width=150, caption=f"**{movie_title}** (Movie ID: {movieId}) - Predicted Score: {score:.2f}")
                # st.badge(movie_title)
                if st.button(f"View Details", key=f"details_{movieId}"):
                    st.session_state['logged_in'] = True
                    st.session_state['userId'] = userId
                    st.session_state['movieId'] = movieId
                    st.session_state['imdb'] = imdb
                    st.switch_page("pages/movie_page.py")

            
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}")
        st.error("Sorry, we couldn't fetch recommendations at this time.")
    st.write("This dashboard is a work in progress. More features coming soon! :construction:")
