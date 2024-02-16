import streamlit as st
from stqdm import stqdm

#Librerías de tiempo, horas y fecha
import time
from time import sleep
from datetime import datetime

#Librerías de dataframes y arrays
import numpy as np
import pandas as pd
import random

#Librerías para interactuar con Spotify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import MemoryCacheHandler
from bokeh.models.widgets import Div
from bs4 import BeautifulSoup

# Modelos de Machine Learning y Normalizacion
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

#Por ver si estas se mantienen
import requests
import json
import pickle
import os
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Importar las funciones de los procesos
from funciones import batch_generator, download_songs_to_recomend, download_audio_features, request_user_pick, flatten_list, songs_clustering, user_pick_labeling, songs_to_playlist, post_playlist, success_playlist, get_music_genres, users_pick_html 
#from eda_funciones import popularidad_artista, popularidad_genero, genero_count, histograma_duration_ms, popularidad_playlist, popularidad_fecha_genero, line_polar_playlist, top_artists_plot
from eda_funciones import eda_ui


def main_page():


    if 'spotify' in st.session_state:
        spotify = st.session_state.spotify

    REDIRECT_URI = st.secrets['SPOTIFY_REDIRECT_URI']
    USERNAME = st.secrets['SPOTIFY_USERNAME']

    st.header(body="App 💻")

    st.write("Welcome to the **Spytify Website** made with **Streamlit**.")

    st.markdown("""The data for this project comes from 
                    [Spotify](https://spotify.com/), using the [Spotify API Documentation](https://developer.spotify.com/)
                    to query on demand all the data we need to process your requests and create the coolest playlist ever for you🤘🏽.""")

    st.write("""To use this app just go to the `User Input` section to request the data that we will use to build
                  models and create your playlist.""")

    st.write("""To use the `Playlist Export` section you should follow the whole process Input -> Recommendations -> Data Analysis.""")

    tab1, tab2, tab3, tab4 = st.tabs(['User Input', 'Clustering Recommendations','Exploratory Data Analysis', 'Playlist Export'])

    with tab1: #User Input
        st.title('User Input')

        st.markdown('**To start the process** you have to choose **1. A track or artist** to query, **2. write a query** of your preference, and **3. choose any genre/genres** you like.')

        with st.form('User Input'):
            query_type = st.radio(label='1. Select your query type',
                        options=('Track', 'Artist'),
                        index=0,
                        disabled=False,
                        horizontal=True,
                        ).lower()
            
            query_text = st.text_input(label= 'Write your query',
                        max_chars=100,
                        placeholder='Track or artist query:')
            
            limit = st.number_input(label='Enter how many results you want on your track query (1-50). If your query type is an artist please ignore this field',
                        min_value=1,
                        max_value=50,
                        value=20,
                        step=1)

            genre = st.multiselect('Select the genre(s) that you want to lookup data',
                        get_music_genres(),
                        default=[],
                        key='genres_multiselect',
                        max_selections=5,
                        placeholder="rock, rap, hip hop"
            )

            submitted = st.form_submit_button("Submit")

            if submitted:

                if query_type == 'track':
                    query_text = 'Outside' if query_text == '' else query_text

                elif query_type == 'artist':
                    query_text = 'Calvin Harris' if query_text == '' else query_text

                genre = ['commercial'] if genre == [] else genre

                st.write(f'Query type: {query_type}   Query: {query_text}   Genre(s): {genre}')
                st.info(f'We are processing your {query_type} requests, please wait. This may take a while...')
                df_user_pick = request_user_pick(spotify, query = query_text, query_type = query_type, limit = limit)

                if len(df_user_pick) > 0:
                    st.success('Your request was succesfully done')
                    #Esto deja la tabla en formato markdown y muestra las columnas con los previews de las canciones
                    #st.markdown(df_user_pick.to_html(render_links=True, escape= False),unsafe_allow_html=True)
                    
                    #Esto muestra el dataframe como siempre
                    st.dataframe(df_user_pick[['track_name', 'track_artists', 'track_popularity']], width = 1000)

                    st.info('We are processing your genre(s) requests, please wait. This may take a while')

                    df = download_songs_to_recomend(genre, spotify)

                    df_songs_to_recomend = download_audio_features(df, spotify)

                    if len(df_songs_to_recomend) > 0:
                        st.success('Your request was succesfully done')
                        st.dataframe(df_songs_to_recomend[['track_name', 'track_artists', 'track_popularity', 'track_album_name', 'track_album_release_date', 'playlist_name', 'playlist_genre']])

                        st.session_state['df_user_pick'] = df_user_pick
                        st.session_state['df_songs_to_recomend'] = df_songs_to_recomend


    with tab2: #Clustering Recommendations

        st.title('Clustering Recommendations')
        st.markdown(f'**To start the clustering process** you have to choose **a track or a list of them** from your initial {query_type} query.\nIf you are not 100% sure of the song you request you can listen :sound: to the results :cd:.')

        if 'df_user_pick' in st.session_state:
            df_user_pick = st.session_state['df_user_pick']
            songs_options = [f'{idx}. {song}' for idx, song in zip(df_user_pick.index, df_user_pick['track_name'])]
            
            #Se crea una tabla formato HMTL con información de las canciones del usuario, con preview de audio e imagen del album disponible
            table_html = users_pick_html(df_user_pick)

            #Mostrar la tabla HTML personalizada
            st.markdown(table_html, unsafe_allow_html=True)
            st.markdown("")

        else:
            songs_options = []
        
        if 'df_songs_to_recomend' in st.session_state:
            df_songs_to_recomend = st.session_state['df_songs_to_recomend']
                    
        with st.form('Index list'):
            
            chosen_songs = st.multiselect(label="Select the songs that you want to make the clustering process", 
                                        options=songs_options, 
                                        default=songs_options[:5])


            audio_features = ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode','speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
            
            audio_features_to_cluster = st.multiselect(label="Select the audio features you want to use in the songs analysis", 
                                                        options=audio_features, 
                                                        default=audio_features)

            clustering_submit = st.form_submit_button("Submit")

            if clustering_submit:
                st.info('The clustering process had started. This may take a while...')

                #index_list = flatten_list(chosen_index)
                index_list = [int(idx.split('. ')[0]) for idx in chosen_songs]

                df_user_pick_to_cluster = df_user_pick.copy()
                df_user_pick_to_cluster = df_user_pick_to_cluster[df_user_pick_to_cluster.index.isin(index_list)]

                user_pick, songs_to_recomend, centroid_label, epsilon = songs_clustering(df_songs_to_recomend, df_user_pick_to_cluster, audio_features=audio_features_to_cluster)

                st.success(f"The clustering process is completed, now you can review your election's group/label.")
                st.dataframe(user_pick[['track_name', 'track_artists', 'track_popularity', 'Label']], width = 800)

                st.session_state['user_pick'] = user_pick
                st.session_state['songs_to_recomend'] = songs_to_recomend
                st.session_state['centroid_label'] = centroid_label


    with tab3: # Exploratory Data Analysis

        if 'songs_to_recomend' in st.session_state:
            songs_to_recomend = st.session_state['songs_to_recomend']

            eda_ui(songs_to_recomend)


    with tab4: # Playlist Export

        songs_limit = 10_000 #Max capacity for a playlist in Spotify
        user_pick_count = 0
        if 'user_pick' in st.session_state:
            user_pick = st.session_state['user_pick']
            user_pick_count = len(user_pick)
        if 'songs_to_recomend' in st.session_state:
            songs_to_recomend = st.session_state['songs_to_recomend']
            songs_limit = len(songs_to_recomend) if len(songs_to_recomend) < songs_limit else songs_limit
        if 'centroid_label' in st.session_state:
            centroid_label = st.session_state['centroid_label']

        st.title('Playlist Export')
        st.markdown(f'Yay! we are almost done **to create your own Spotify playlist** you have to choose the configuration you prefer for it')

        with st.form('Playlist'):

            name = st.text_input(label= 'Name your playlist',
                        max_chars=100,
                        placeholder='Playlist name')
            
            description = st.text_input(label= 'Add a description to your playlist ',
                        max_chars=100,
                        placeholder='Playlist description')

            combine_option = st.radio(label='Do you want to include your picks in the playlist?',
                        options=('Yes', 'No'),
                        index=0,
                        disabled=False,
                        horizontal=True,
                        ).lower()
            
            popularity_option = st.radio(label='Which recommended songs do you want to include in your playlist?',
                        options=('Most popular', 'Less popular'),
                        index=0,
                        disabled=False,
                        horizontal=True,
                        ).lower()
                        
            shuffle_option = st.radio(label='Do you want to order your playlist songs randomly? Otherwise the most/less popular songs will be picked',
                        options=('Yes', 'No'),
                        index=1,
                        disabled=False,
                        horizontal=True,
                        ).lower()
            
            songs_number = st.slider(label='How many songs do you want to add to your playlist?',
                                    min_value=(user_pick_count if user_pick_count > 0 else 1),
                                    max_value=(songs_limit + user_pick_count),
                                    value=100)
                            
            playlist_submit = st.form_submit_button("Submit")

            if playlist_submit:
    
                combine = True if combine_option == 'yes' else False
                popularity = False if popularity_option == 'most popular' else True
                shuffle = True if shuffle_option == 'yes' else False

                songs_to_recomend = songs_to_recomend[songs_to_recomend['Label'] == centroid_label].sort_values(by = 'track_popularity', ascending = popularity)

                songs_to_playlist_ids = songs_to_playlist(user_pick, songs_to_recomend, combine, songs_number, shuffle)

                playlist_url, playlist_name, number_songs_uploaded = post_playlist(spotify, REDIRECT_URI, USERNAME, songs_to_playlist_ids, name, description)

                if number_songs_uploaded > 0:

                    success_playlist(playlist_url, playlist_name, number_songs_uploaded)

if __name__ == "__main_page__":
    main_page()
