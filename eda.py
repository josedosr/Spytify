import streamlit as st

#Librerías de dataframes y arrays
import numpy as np
import pandas as pd

#Librerías para interactuar con Spotify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import MemoryCacheHandler
from bokeh.models.widgets import Div

#Importar las funciones de los procesos
from funciones import get_all_playlist_items, user_playlist_stats_request, batch_generator
from eda_funciones import eda_ui, eda_playlist_query, popularidad_artista, popularidad_genero, genero_count, histograma_duration_ms, popularidad_playlist, popularidad_fecha_genero, line_polar_playlist, top_artists_plot


def eda():

    st.title('Exploratory Data Analysis')

    st.markdown(f'In this section you can visualize the Exploratory Data Analysis for six from the most popular musical genres, or you can generate a new DataFrame with data requested from a playlist from your choice and analyze it')

    tab1, tab2 = st.tabs(['Exploratory Data Analysis', 'Your Playlist Data Analysis'])

    with tab1:

        if 'df_recommendations_cache' in st.session_state:
            df_recommendations_cache = st.session_state['df_recommendations_cache']
            
            with st.expander(label="DataFrame", expanded=False):
                st.dataframe(df_recommendations_cache[['track_name', 
                                                       'track_artists', 
                                                       'track_popularity', 
                                                       'track_album_name', 
                                                       'track_album_release_date',
                                                       'playlist_name', 
                                                       'playlist_genre']])

            eda_ui(df_recommendations_cache)


    with tab2:

        if 'spotify' in st.session_state:
            spotify = st.session_state['spotify']
        
        with st.form('User Input'):

            playlist_url = st.text_input(label= 'Write your playlist id / playlist url / playlist shared link',
                        max_chars=200,
                        placeholder='Playlist')
    
            playlist_submitted = st.form_submit_button("Submit")

            if playlist_submitted:
                df_user_playlist_stats_request = user_playlist_stats_request(spotify, playlist_url)
        
                if len(df_user_playlist_stats_request) > 0:
                    with st.expander(label="DataFrame", expanded=False):
                        st.dataframe(df_user_playlist_stats_request[['track_name', 'track_artists', 'track_popularity']], width = 800)

                    eda_playlist_query(df_user_playlist_stats_request)

if __name__ == "__eda__":
    eda()
