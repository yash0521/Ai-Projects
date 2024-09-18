import pandas as pd
import requests
import  streamlit as st
import pickle

similarity = pickle.load(open("similarity.pkl", "rb"))
movies = pickle.load(open("movies_dictionary.pkl", "rb"))
df_movies = pd.DataFrame(movies)

def fetch_poster(movie_id):
    api_key= "865ce4288d4f0e3704812c24a0ff58b2"
    url = "https://api.themoviedb.org/3/movie/{}?api_key={}".format(movie_id, api_key)
    response = requests.get(url)
    data = response.json()
    # st.text(url)
    return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]


def recommend(movie):
    movie_index = df_movies[df_movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])

    recommended_movies = []
    recommended_movies_posters = []
    for i in distances[1:6]:  # first 5 movies
        recommended_movies.append(df_movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(df_movies.iloc[i[0]].movie_id))

    return recommended_movies, recommended_movies_posters

st.title("Movie Recommended System")

selected_movie_name = st.selectbox(
    'Enter Movie name ?',
    df_movies["title"].values)

if st.button('Recommend'):
    name, posters = recommend(selected_movie_name)
    import streamlit as st

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(name[0])
        st.image(posters[0])

    with col2:
        st.text(name[1])
        st.image(posters[1])

    with col3:
        st.text(name[2])
        st.image(posters[2])

    with col4:
        st.text(name[3])
        st.image(posters[3])

    with col5:
        st.text(name[4])
        st.image(posters[4])
