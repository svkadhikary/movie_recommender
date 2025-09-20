import os
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict
import pandas as pd
import streamlit as st
from utils.get_poster import get_image, get_description
from utils import links_helper, movie_helper, ratings_helper, user_helper
from recommenders.prediction import Prediction

from logging_custom.logger import Logger

load_dotenv()

# Set page configuration
st.set_page_config(page_title="Movie Recommender App", layout="wide")

# Initialize logger
logger = Logger("movie_page_logger").get_logger()

# if new movie selected
if 'pending_movieId' in st.session_state:
    st.session_state['movieId'] = st.session_state.pop('pending_movieId')
    st.session_state['imdb'] = st.session_state.pop('pending_imdb')
    st.rerun()

# go dashboard if rated
if st.session_state.get('go_dashboard', False):
    logger.info("Redirecting to dashboard")
    st.session_state['go_dashboard'] = False
    st.switch_page('pages/dashboard.py')

# Redirect to home if not logged in
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.session_state['logged_in'] = False
    st.switch_page("app.py")
# get user id
userId = st.session_state.get('userId', None)
# get movie id and imdb id
movieId = st.session_state.get('movieId', None)
# get imdb id
imdb = st.session_state.get('imdb', None)
logger.info(f"Displaying details for movieId {movieId} and IMDb ID {imdb} for userId {userId}.")

# # Initialize helper functions
links_helper = links_helper.LinksHelper()
movie_helper = movie_helper.MovieHelper()
user_helper = user_helper.UserHelper()
ratings_helper = ratings_helper.RatingsHelper()

prediction = Prediction(user_helper, movie_helper, links_helper, ratings_helper)
# update ratings
def on_rating_change():
    rating = st.session_state.get('rating_input')
    movieId = st.session_state.get('movieId')
    userId = st.session_state.get('userId')
    # save directly in dataset
    if rating is not None and movieId is not None and userId is not None:
        ts = datetime.timestamp(datetime.now())
        ratings_path = "datasets/ml-latest/new_ratings.csv"
        new_row = {
            'userId': userId,
            'movieId': movieId,
            'rating': rating,
            'timestamp': int(ts)
        }
        try:
            if os.path.exists(ratings_path):
                df = pd.read_csv(ratings_path)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df = pd.DataFrame([new_row])
            df.to_csv(ratings_path, index=False)
            logger.info(f"Saved new rating to {ratings_path}")
        except Exception as e:
            logger.error(f"Error saving rating: {e}")
            st.error("Failed to save rating.")
    else:
        st.error("Missing rating, movieId, or userId.")
    

@st.cache_data
def get_similar_movies_cached(movieId):
    return prediction.cmf_simlar_movies(int(movieId))

# initialize columns
col1, col2 = st.columns([4, 1])
# logout button
with col2:
    # Logout button
    if st.button("Logout", key="logout_button"):
        st.session_state.update({'logged_in': False,
                                 'userId': None,
                                 'movieId': None,
                                 'imdb': None})
        st.switch_page("app.py")
# page content
with col1:
    st.title(f":grey[Movie Details] :movie_camera:")
    col1_1, col1_2 = st.columns([1, 2])
    with col1_1:
        st.image(get_image(imdb, movieId, links_helper), width=200)
        # st.badge(label=str(movieId), icon=":material/check:")

    with col1_2:
        st.write(f"Movie ID: {movieId}")
        st.write(f"{movie_helper.movies_df[movie_helper.movies_df['movieId'] == movieId]['title'].values[0]}")
        st.text(get_description(imdb))
        st.write(f"Genres: {movie_helper.movies_df[movie_helper.movies_df['movieId'] == movieId]['genres'].values[0]}")
        
        # st.selectbox("Rate this movie", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], key="rating_input", on_change=on_rating_change)
        cols = st.columns(4)
        # Rating input
        with cols[0]:
            rating = st.number_input(
                "Rate this movie",
                min_value=1.0,
                max_value=5.0,
                step=0.5,
                format="%.1f",
                key="rating_input"
            )
            # Submit button
            if st.button("Submit Rating"):
                if on_rating_change():
                    st.success("Rating submitted!")
                # Get ratings dict from session state
                # new_ratings = st.session_state.get('new_ratings', defaultdict(list))
                # if new_ratings:
                #     # # Convert to DataFrame
                #     # df = pd.DataFrame(new_ratings)
                #     # # Append to CSV (create if not exists)
                #     # df.to_csv("new_ratings.csv", mode='a', header=not pd.io.common.file_exists("new_ratings.csv"), index=False)
                #     # on_rating_change function
                # else:
                #     st.warning("No rating data found.")
        # user rating
        with cols[1]:
            try:
                user_rating = ratings_helper.ratings_df[(ratings_helper.ratings_df['userId'] == int(userId)) & (ratings_helper.ratings_df['movieId'] == int(movieId))]['rating'].values[0]
            except Exception as e:
                user_rating = "You haven't rated this movie yet."
            st.write(f"**Your Rating: {user_rating}**")
        # average rating
        with cols[2]:
            st.write(f"**Average Rating: {ratings_helper.ratings_df[ratings_helper.ratings_df['movieId'] == int(movieId)]['rating'].mean()}**")

    st.subheader("Movies similar to this one:")
    try:
        similar_movies = get_similar_movies_cached(movieId)
        logger.info(f"Number of similar movies found: {len(similar_movies)}")
        similar_movies = [(movieId, similarity, links_helper.get_imdb_id(movieId)) for movieId, similarity in similar_movies]

        for i, (movieId, similarity, imdb_id) in enumerate(similar_movies[:12]):
            if i % 4 == 0:
                cols = st.columns(4)

            with cols[i % 4]:
                movie_title = movie_helper.movies_df[movie_helper.movies_df['movieId'] == movieId]['title'].values[0]
                st.image(get_image(imdb_id, movieId, links_helper), width=150, caption=f"**{movie_title}** (Movie ID: {movieId}) - Similarity: {similarity:.2f}")
                # st.badge(movie_title)
                # movie to show
                view_details = st.button(f"View Details", key=f"details_{movieId}")
                if view_details:
                    logger.info(f"Redirecting to selected movie {movieId} page")
                    st.session_state['logged_in'] = True
                    st.session_state['userId'] = userId
                    st.session_state['pending_movieId'] = movieId
                    st.session_state['pending_imdb'] = imdb_id

    except Exception as e:
        logger.error(f"Error fetching recommendations: {str(e)}")
        st.error("Sorry we couldn't fetch recommendations at this time")

