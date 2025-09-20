import os
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import numpy as np
from logging_custom.logger import Logger
from utils import movie_helper, links_helper, user_helper
from utils.get_poster import get_image, get_random_image

load_dotenv()

logger = Logger("New_User").get_logger()

movie_helper = movie_helper.MovieHelper()
links_helper = links_helper.LinksHelper()

try:
    # create new user ID, and save in session
    if 'userId' not in st.session_state or not st.session_state['userId']:
        user_helper = user_helper.UserHelper()
        # user_helper.users_df.userId = user_helper.users_df.userId.astype(int)
        userId = user_helper.users_df.index.max() + 1
        logger.info(f"New User Id: {userId}")
        del(user_helper)
        st.session_state['userId'] = userId
    if 'new_ratings' not in st.session_state:
        # initialize empty dict for collecting ratings
        st.session_state['new_ratings'] = {
            'userId': [],
            'movieId': [],
            'rating': [],
            'timestamp': []
            }
except Exception as e:
    raise e
# if new rating dict in session state
new_ratings = st.session_state.get("new_ratings")

# Search movies
# def search_movies(search_query):
#     return movie_helper.movies_df[movie_helper.movies_df['title'].str.contains(search_query, case=False, na=False)]

# with st.sidebar:
#     search = st.text_input("Search movies", key='search_movies', placeholder='Avengers...')
#     if search:
#         results = search_movies(search)
#         for _, row in results.iterrows():
#             if st.button(row['title'], key=f"search_{row['movieId']}"):
#                 st.session_state['movieId'] = int(row['movieId'])
#                 st.session_state['imdb'] = links_helper.get_imdb_id(row['movieId'])

col_1, col_2 = st.columns([4, 1])

with col_2:
    if st.button("Refresh Movies list"):
        st.session_state['random_state'] = np.random.randint(0, 100000)

with col_1:
    try:
        st.header("Welcome to Movie Recommender :movie_camera:")
        st.subheader("Please like at least 3 movies to get recommendations :heart:")
        rand_state = st.session_state.get('random_state', 42)
        random_movies = movie_helper.get_random_movies(40, rand_state)
        userId = st.session_state['userId']

        for i, (_, row) in enumerate(random_movies.iterrows()):
            if i % 4 == 0:
                cols = st.columns(4)

            with cols[i % 4]:
                movie_id, movie_title = row['movieId'], row['title']
                imdb_id = links_helper.get_imdb_id(movie_id)
                # st.image(get_image(imdb_id, movie_id, links_helper), width=150, caption=f"**{movie_title}** (Movie ID: {movie_id})")
                # st.image(get_random_image("H:/mobile_backup/Download"), width=150, caption=f"**{movie_title}** (Movie ID: {movie_id})")
                st.image("content/poster-placeholder.webp", width=150, caption=f"**{movie_title}** ({movie_id})")

                if st.button(":star: Like", key=f"like{movie_id}"):
                    st.balloons()
                    ts = int(datetime.timestamp(datetime.now()))
                    
                    new_ratings['userId'].append(userId)
                    new_ratings['movieId'].append(movie_id)
                    new_ratings['rating'].append(5)
                    new_ratings['timestamp'].append(ts)
                    st.session_state['new_ratings'] = new_ratings
                    
        if len(new_ratings['movieId']) >= 3:
            st.success("You've liked 3 or more movies! You can now get recommendations.")
            if st.button("Get Recommendations"):
                st.session_state['new_ratings'] = new_ratings
                logger.info(f"New ratings data in session state: {new_ratings}")
                st.switch_page("pages/cold_start_rec.py")

    except Exception as e:
        logger.error(f"Can find movies. Error: {e}")
        raise e

