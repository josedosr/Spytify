import streamlit as st
import os

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
from bokeh.models.widgets import Div

#Librerías que dan estructura a la web
from home import app_display_welcome, app_get_token, get_token, app_sign_in, sign_in#, get_spotify_code
from process_description import process_description
from main_page import main_page
from eda import eda
from create_playlist import create_playlist
from about import about

PAGE_CONFIG = {'page_title': 'Spytify',
               'page_icon': '🐍',
               'layout': 'wide',
               'initial_sidebar_state': 'expanded'}

st.set_page_config(**PAGE_CONFIG)

#Credentials
CLIENT_ID = st.secrets['SPOTIFY_CLIENT_ID']
CLIENT_SECRET = st.secrets['SPOTIFY_CLIENT_SECRET']
REDIRECT_URI = st.secrets['SPOTIFY_REDIRECT_URI']
SCOPE = st.secrets['SPOTIFY_SCOPE']
USERNAME = st.secrets['SPOTIFY_USERNAME']

@st.cache_data
def load_data():
    #Carga del DataFrame
    df = pd.read_csv('sources/songs_to_recomend.csv')
    return df

df_recommendations_cache = load_data()
df_recommendations_cache['track_artists'] = df_recommendations_cache['track_artists'].apply(lambda x : eval(x) if pd.notna(x) else []) #Evalua la lista que está en la columna del .csv
df_recommendations_cache["year"] = df_recommendations_cache["track_album_release_date"].apply(lambda x: int(x[:4]))

# Guarda el dataframe en la sesión de usuario
if 'df_recommendations_cache' not in st.session_state:
    st.session_state['df_recommendations_cache'] = df_recommendations_cache

@st.cache_data
def load_user_query():
    #Carga del DataFrame
    df = pd.read_csv('sources/user_query.csv')
    df.index = df.index + 1
    return df

df_user_query = load_user_query()
df_user_query['track_artists'] = df_user_query['track_artists'].apply(lambda x : eval(x) if pd.notna(x) else []) #Evalua la lista que está en la columna del .csv

# Guarda el dataframe en la sesión de usuario
if 'df_user_query' not in st.session_state:
    st.session_state['df_user_query'] = df_user_query

@st.cache_data
def load_epsilon():
    #Carga del DataFrame
    df = pd.read_csv('sources/epsilon.csv')
    return df

epsilon = load_epsilon()

# Guarda el dataframe en la sesión de usuario
if 'epsilon' not in st.session_state:
    st.session_state['epsilon'] = epsilon




def main():
    
    menu = ['Home', 'Process Description', 'Main App', 'Exploratory Data Analysis', 'Create your playlist', 'About']

    page = st.sidebar.selectbox(label="Menu", options=menu)

    if page == 'Home':

        # Mostrar la bienvenida y obtener el enlace de autorización
        app_display_welcome(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, SCOPE, USERNAME)
        
        app_get_token()
            
        # Verificar si hay un token de acceso en la sesión
        if "cached_token" in st.session_state:
            # Si hay un token de acceso, iniciar sesión y crear el objeto Spotify
            spotify = app_sign_in()

            st.session_state['spotify'] = spotify
            
        pass

    elif page == 'Process Description':
        
        process_description()

        pass

    elif page == 'Main App':
        
        main_page()

        pass

    elif page == 'Exploratory Data Analysis':
        
        eda()

        pass

    elif page == 'Create your playlist':

        create_playlist()

        pass

    elif page == 'About':

        about()

        pass

    else:

        pass



if __name__ == '__main__':
    main()
