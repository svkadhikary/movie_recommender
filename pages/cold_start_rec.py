import os
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import pandas as pd
from logging_custom.logger import Logger
from utils import links_helper, user_helper, movie_helper
from utils.get_poster import get_image, get_random_image
from recommenders.cold_start import ColdStartRecommender

movie_helper = movie_helper.MovieHelper()
links_helper = links_helper.LinksHelper()
user_helper = user_helper.UserHelper()
cold_start = ColdStartRecommender(movie_helper)

load_dotenv()
NEW_RATINGS_DATA = os.getenv("NEW_RATINGS_DATA")

logger = Logger("Cold_Start_rec").get_logger()

# check for user Id
try:
    if 'userId' in st.session_state or st.session_state['userId']:
        # get user id from session state
        userId = st.session_state.get('userId', None)
        logger.info(f"User ID: {userId}")
        # get new user ratings
        new_ratings = st.session_state.get('new_ratings', None)
        logger.info(f"User ratings data: {new_ratings}")
    else:
        logger.error("No user ID found")
except Exception as e:
    raise ValueError(f"{e}")
# get recommendations for user
def predict_recommendation(user_data, model_option):
    try:
        if model_option == 'User Genre Similarity Based':
            recommendation, user_vector = cold_start.recommend_from_liked(user_data['movieId'], top_n=8, threshold=0.8)
            recommendation = [(mid, links_helper.get_imdb_id(mid)) for mid in recommendation]
        else:
            recommendation = cold_start.xgb_cold_start(pd.DataFrame(user_data))
            user_vector = cold_start.get_user_preference_vector(user_data['movieId'], user_data['rating'])
        logger.info(f"Got recommendations using {model_option}: {recommendation}")
        return recommendation, user_vector
    except Exception as e:
        logger.error(f"Failed to get recommendations {e}")
        raise e

with st.sidebar:
    model_option = st.selectbox("Choose Recommendation Model", ("XGBoost Recommender", "User Genre Similarity Based"))
# prediction as per model chosen
recommendation, user_vector = predict_recommendation(new_ratings, model_option)

col_1, col_2 = st.columns([4, 1])
# logout button
with col_2:
    st.write("To update your ratings and get model predictions, go to Home and **Update Model**")
    if st.button("Home", key="home"):
        # update new ratings csv
        df_new_ratings = pd.read_csv(NEW_RATINGS_DATA)
        df_new_ratings = pd.concat([df_new_ratings, pd.DataFrame(new_ratings)], ignore_index=True)
        df_new_ratings.to_csv(NEW_RATINGS_DATA, index=False)
        logger.info(f"New ratings data updated")

        # Clear session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.switch_page("app.py")
    st.write("User Preferences")
    user_vector = user_vector.sort_values(by='Genre score', ascending=False)
    st.dataframe(user_vector.style.background_gradient(cmap='summer', axis=None), height=600)

with col_1:
    st.subheader(f"Movies similar to user {userId}'s preference:: **{model_option}**")

    for i, (movieId, imdb_id) in enumerate(recommendation[:20]):
        if i % 4 == 0:
            cols = st.columns(4)

        with cols[i % 4]:
            movie_title = movie_helper.movies_df[movie_helper.movies_df['movieId'] == movieId]['title'].values[0]
            # st.image(get_random_image(location="H:/mobile_backup/Download"), width=150, caption=f"**{movie_title}** (Movie ID: {movieId})")
            st.image("content/poster-placeholder.webp", width=150, caption=f"**{movie_title}** ({movieId})")
            # if st.button(":star: Like", key=f"like{movieId}"):
            #         st.balloons()
            #         ts = int(datetime.timestamp(datetime.now()))

            #         new_ratings['userId'].append(userId)
            #         new_ratings['movieId'].append(movieId)
            #         new_ratings['rating'].append(5)
            #         new_ratings['timestamp'].append(ts)

            #         st.session_state['new_ratings'] = new_ratings
            #         logger.info(f"User ratings updated: {new_ratings}")

st.write("Note down the **User ID**. *Train* the model and login with the **User ID**   :sunglasses:")


