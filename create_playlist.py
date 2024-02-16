import streamlit as st

#Librerías de tiempo, horas y fecha
import time
from time import sleep
from datetime import datetime

#Librerías de dataframes y arrays
import numpy as np
import pandas as pd

#Librerías para interactuar con Spotify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import MemoryCacheHandler

#Importar las funciones de los procesos
from funciones import batch_generator, songs_to_playlist, post_playlist, success_playlist, get_songs_to_playlist


def create_playlist():

    REDIRECT_URI = st.secrets['SPOTIFY_REDIRECT_URI']
    USERNAME = st.secrets['SPOTIFY_USERNAME']

    if 'spotify' in st.session_state:
            spotify = st.session_state['spotify']

    st.title('Create Playlist')

    st.markdown(f'In this section you can create a playlist with a list of your favorite artists, or you can generate your playlist uploading a .csv file with the songs you want to include')

    tab1, tab2 = st.tabs(['Your Playlist from .csv', 'Your Playlist from Artists input'])

    with tab1:
        st.info('Upload a .csv file in the sidebar')

        st.session_state['csv'] = True

        if st.session_state.csv:
            st.sidebar.markdown("*"*10)
            uploaded_file = st.sidebar.file_uploader(label = "Upload your input CSV file", type = ["csv"])
            st.sidebar.markdown("*"*10)

            if uploaded_file is not None:

                st.success(f'{uploaded_file.name} was successfully uplodaded')

                uploaded_songs = pd.read_csv(filepath_or_buffer = uploaded_file)

                uploaded_songs = uploaded_songs.drop([column for column in uploaded_songs.columns if column != 'track_name'], axis = 1)

                if len(uploaded_songs) > 1:

                    with st.form('File Input'):

                        name = st.text_input(label= 'Name your playlist',
                                    max_chars=100,
                                    placeholder='Playlist name')
                        
                        description = st.text_input(label= 'Add a description to your playlist ',
                                    max_chars=100,
                                    placeholder='Playlist description')
                        
                        file_submitted = st.form_submit_button("Submit")

                        if file_submitted:

                            uploaded_songs_to_playlist = uploaded_songs['track_name'].to_list()

                            df_songs_to_playlist, songs_number = get_songs_to_playlist(spotify, items = uploaded_songs_to_playlist, query_type = 'track')
                            songs_to_playlist_ids = songs_to_playlist(None, df_songs_to_playlist, combine = False, songs_number = songs_number, shuffle = True)
                            playlist_url, playlist_name, number_songs_uploaded = post_playlist(spotify, REDIRECT_URI, USERNAME, songs_to_playlist_ids, name, description)

                            if number_songs_uploaded > 0:

                                success_playlist(playlist_url, playlist_name, number_songs_uploaded)


    with tab2:
        
        with st.form('Artist List Input'):

            name = st.text_input(label= 'Name your playlist',
                        max_chars=100,
                        placeholder='Playlist name')
            
            description = st.text_input(label= 'Add a description to your playlist ',
                        max_chars=100,
                        placeholder='Playlist description')

            artist_list = st.text_input(label= 'Write your favorites artists separated with commas',
                        max_chars=200,
                        placeholder='Artists')
    
            artists_submitted = st.form_submit_button("Submit")

            if artists_submitted:

                artists = artist_list.replace(' ,',',').replace(', ',',')
                artists = artists.split(',')
                artists = [artist for artist in artists if artist != '']
                artists = ['Milo J'] if artists == [] else artists

                df_songs_to_playlist, songs_number = get_songs_to_playlist(spotify, items = artists, query_type = 'artist')
                songs_to_playlist_ids = songs_to_playlist(None, df_songs_to_playlist, combine = False, songs_number = songs_number, shuffle = True)
                playlist_url, playlist_name, number_songs_uploaded = post_playlist(spotify, REDIRECT_URI, USERNAME, songs_to_playlist_ids, name, description)

                if number_songs_uploaded > 0:

                    success_playlist(playlist_url, playlist_name, number_songs_uploaded)

                    







if __name__ == "__create_playlist__":
    create_playlist()
