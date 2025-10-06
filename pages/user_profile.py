import streamlit as st
from logging_custom.logger import Logger
from utils import movie_helper, user_helper, links_helper, ratings_helper
from utils.get_poster import get_image
from recommenders.cold_start import ColdStartRecommender


# Redirect to home if not logged in
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.session_state['logged_in'] = False
    st.switch_page("app.py")

# Initialize logger
logger = Logger("User_Profile_page").get_logger()
# get user id
userId = st.session_state.get('userId', None)
# Set page configuration
st.set_page_config(page_title="Movie Recommender App", layout="wide")

# initialize helper classes
movie_helper = movie_helper.MovieHelper()
user_helper = user_helper.UserHelper()
links_helper = links_helper.LinksHelper()
ratings_helper = ratings_helper.RatingsHelper()

col1, col2 = st.columns([4, 1])
with col2:
    # Logout button
    st.button("Logout", on_click=lambda: st.session_state.update({'logged_in': False, 'userId': None}))

with col1:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)
    st.title(f":grey[{userId}'s profile] :movie_camera:")
    st.write("Welcome to the Movie Recommender App! :star:")
    st.write("Top 5 Genres as per your ratings:")
    try:
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
        cols_i = st.columns(5)
        # loop and show genres
        i = 0
        for genre, row in user_vec_df.head().iterrows():
            with cols_i[i]:
                # st.write(f"**{genre}**:{row['Genre score']}")
                st.metric(label=f"**{genre}**", value=f"{row['Genre score']}")
                i += 1
    except Exception as e:
        logger.error(f"Error fetching user vector::: {e}")
        st.error(f"Error fetching user preferences: {e}")
    
    try:
        # top rated movies
        user_profile_top_10 = user_profile.sort_values(by='rating', ascending=False).head(10).reset_index(drop=True)
        # get imdb ids of the movies
        user_profile_top_10['imdbId'] = [links_helper.get_imdb_id(movieId) for movieId in user_profile_top_10['movieId'].values]
        st.write("Top 10 movies rated by you")
        for j, row in user_profile_top_10.iterrows():
            if j % 5 == 0:
                cols_j = st.columns(5)
            
            with cols_j[j % 5]:
                # get movies title
                movie_title = movie_helper.movies_df[movie_helper.movies_df['movieId'] == row['movieId']]['title'].values[0]
                # show movie
                st.image(get_image(row['imdbId'], row['movieId'], links_helper), width=150, caption=f"**{movie_title}** (Movie ID: {row['movieId']}) - Your Rating: {row['rating']:.2f}")
                # view movie details
                if st.button(f"View Details", key=f"details_{row['movieId']}"):
                        st.session_state['logged_in'] = True
                        st.session_state['userId'] = userId
                        st.session_state['movieId'] = row['movieId']
                        st.session_state['imdb'] = row['imdbId']
                        st.switch_page("pages/movie_page.py")
        if st.button("See all", key="see_all"):
            # See all movies rated by user
            user_profile_rest = user_profile.sort_values(by='rating', ascending=False).reset_index(drop=True)
            user_profile_rest = user_profile_rest.loc[10:]
            user_profile_rest['imdbId'] = [links_helper.get_imdb_id(movieId) for movieId in user_profile_rest['movieId'].values]
            for j, row in user_profile_rest.iterrows():
                if j % 5 == 0:
                    cols_j = st.columns(5)
                
                with cols_j[j % 5]:
                    # get movies title
                    movie_title = movie_helper.movies_df[movie_helper.movies_df['movieId'] == row['movieId']]['title'].values[0]
                    # show movie
                    st.image("content/poster-placeholder.webp", width=150, caption=f"**{movie_title}** (Movie ID: {row['movieId']}) - Your Rating: {row['rating']:.2f}")
                    # view movie details
                    if st.button(f"View Details", key=f"details_{row['movieId']}"):
                            st.session_state['logged_in'] = True
                            st.session_state['userId'] = userId
                            st.session_state['movieId'] = row['movieId']
                            st.session_state['imdb'] = row['imdbId']
                            st.switch_page("pages/movie_page.py")

    except Exception as e:
        logger.error(f"Error fetching user rated movies::: {e}")
        st.error(f"Error fetching user rated movies: {e}")


