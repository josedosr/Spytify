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


def create_spotify_object(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, SCOPE, USERNAME):

    token = SpotifyOAuth(scope         = SCOPE, 
                         username      = USERNAME,
                         client_id     = CLIENT_ID,
                         client_secret = CLIENT_SECRET,
                         redirect_uri  = REDIRECT_URI,
                         cache_handler = MemoryCacheHandler())

    # Se obtiene el token de acceso utilizando la instancia de SpotifyOAuth
    #access_token_info = token.get_access_token()

    # Se crea una instancia de Spotify usando el token de acceso
    spotify_object = spotipy.Spotify(auth_manager = token)

    return spotify_object

def check_token(token, access_token_info):
    
    if token.is_token_expired(access_token_info):
        access_token_info = token.get_access_token()
        changed = True
        
    else:
        changed = False
        
    return access_token_info, changed

def refresh_spotify_object(access_token_info):
    
    # Se crea una instancia de Spotify usando el token de acceso
    spotify_object = spotipy.Spotify(auth_manager = access_token_info['access_token'])

    return spotify_object


def batch_generator(lista, tamaño_lote = 100):
    for i in range(0, len(lista), tamaño_lote):
        yield lista[i:i + tamaño_lote]


def flatten_list(string):

    if not all(char.isdigit() or char in ',- ' for char in string):
        return [1]
    
    try:
        # Limpiar la cadena
        clean_string = string.replace(' ,', ',').replace(', ', ',').replace(' ', '')

        # Dividir la cadena en una lista
        numbers_list = clean_string.split(',')

        # Procesar la lista y aplanarla
        numbers_list = [int(num) if '-' not in num else list(range(int(num.split('-')[0]), int(num.split('-')[-1]) + 1)) for num in numbers_list if num != '']

        # Aplanar la lista resultante
        numbers_list = [num for sublist in numbers_list for num in (sublist if isinstance(sublist, list) else [sublist])]

        numbers_list = [1] if numbers_list == [] else numbers_list

        return numbers_list
    
    except:
        return [1]

        
def user_pick_labeling(row, centroids, audio_features):
    
    row_values = row[audio_features].astype(float).values.reshape(1,-1)
    centroid_values = centroids[audio_features].astype(float).values
    
    distances = cdist(row_values, centroid_values)
    min_distance = distances.argmin()
    # print(min_distance)
    
    closest_label = centroids.loc[min_distance, 'Label']
    
    return closest_label


def download_songs_to_recomend(genres, spotify):
    playlist_limit = 20
    songs_limit = 80
    results = []

    # Itera sobre los géneros que selecciona el usuario y va extrayendo playlists de esos géneros
    for genre in genres:

        playlists = spotify.search(q = genre, limit = playlist_limit, offset = 0, type = 'playlist', market = 'ES')
        playlists = playlists['playlists']['items']

        # Dentro de cada playlist que consigue por cada género itera para extraer las canciones que tenga.
        for playlist in stqdm(playlists, desc=f'Processing playlists of {genre} genre'):

            #playlist data
            playlist_id = playlist['id']
            playlist_name = playlist['name']
            playlist_genre = genre

            # Si la playlist tiene más de 100 canciones se puede establecer que se llame a la API de Spotify para solicitar más canciones dentro de la playlist
            limit = playlist['tracks']['total'] if playlist['tracks']['total'] < songs_limit else songs_limit
            for offset in range(0,limit,100):

                songs = spotify.playlist_tracks(playlist_id = playlist_id, limit=songs_limit, offset = offset, market = 'ES')
                songs = songs['items']

                for idx, song in enumerate(songs):

                    try: 
                        #track data
                        track = song['track']
                        
                        if (track is not None or track != None) and track['is_playable'] == True:

                            track_id = track['id']
                            track_name = track['name']
                            track_artists = [artist['name'] for artist in track['artists']]
                            track_popularity = track['popularity']

                            #album data
                            album = track['album']
                            track_album_id = album['id']
                            track_album_name = album['name']
                            track_album_release_date = album['release_date']

                            results.append([track_id, track_name, track_artists, track_popularity,track_album_id, track_album_name, track_album_release_date,playlist_name, playlist_id, playlist_genre])

                    except:
                        pass   

                sleep(0.2) #Este sleep es para bajar el ritmo de llamadas a la API para sacar las canciones de cada playlist encontrado

        sleep(0.5) #Este sleep es para bajar el ritmo de llamadas a la API para sacar los playlists de cada género

    df = pd.DataFrame(results, columns = ['track_id', 'track_name', 'track_artists', 'track_popularity', 'track_album_id', 'track_album_name', 'track_album_release_date', 'playlist_name', 'playlist_id', 'playlist_genre'])
    df = df.drop_duplicates(['track_id']) # Se hace drop duplicates por si hay canciones que estén repetidas en varios playlists.
    df = df.dropna(axis = 0).reset_index(drop=True)
    
    return df


def download_audio_features(df, spotify):

    songs_audio_features = []

    songs_list = df['track_id'].to_list()

    """
    Dado que la API de Spotify solo permite hacer consultas de hasta 100 elementos en una sola llamada, 
    creamos un generador de batches, en este caso 100 track_id's para poder pasarle a la API conjuntos de canciones.
    """
    batches = list(batch_generator(songs_list)) 

    for batch in stqdm(batches,desc="Getting audio features..."):

        # Llamada a la API de Spotify para descargar los audio features a partir de un batch de hasta 100 elementos
        features_request = spotify.audio_features(batch)

        for song in features_request:

            try:
                track_id = song['id']
                danceability = song['danceability']
                energy = song['energy']
                key = song['key']
                loudness = song['loudness']
                mode = song['mode']
                speechiness = song['speechiness']
                acousticness = song['acousticness']
                instrumentalness = format(song['instrumentalness'], '.10f') #Como los valores de este campo tienen muchos decimales formateamos el valor para tener el número completo
                liveness = song['liveness']
                valence = song['valence']
                tempo = song['tempo']
                duration_ms = song['duration_ms']
                time_signature = song['time_signature']

                songs_audio_features.append([track_id, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, time_signature])

            except:
                pass

        sleep(0.5)

    df_audio_features = pd.DataFrame(songs_audio_features, columns = ['track_id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'])
    df_downloaded = df.copy()
    df_downloaded = df_downloaded.merge(df_audio_features, how = 'inner', left_on = 'track_id', right_on = 'track_id') #Se hace un merge del df de entrada + el df resultado de los audio features que se ha generado. 
    df_downloaded = df_downloaded.dropna(axis = 0).reset_index(drop=True)

    return df_downloaded



def request_user_pick(spotify, query = 'golden', query_type = 'track', limit = 20):

    # Dependiendo del query type se diseñan búsquedas distintas para hacer peticiones a la API de Spotify

    if query_type == 'track':

        """ if limit != 20 and limit <= 1:
            limit = 1
        elif limit != 20 and limit >= 50:
            limit = 50 """

        # Si la query type es de tipo track, se buscan canciones con el input del usuario
        songs = spotify.search(q = query, limit = limit, offset = 0, type = query_type, market = 'ES')
        songs = songs['tracks']['items']

    else:
        
        # Si la query type es de tipo artist, se hace una petición del artista que ha pedido el usuario, luego se con el ID del artista, se hace una petición de las top 10 canciones más populares. 
        limit = 1
        artist = spotify.search(q = query, limit = limit, offset = 0, type = query_type, market = 'ES') 
        artist_id = artist['artists']['items'][0]['id']
        songs = spotify.artist_top_tracks(artist_id = artist_id, country = 'ES')
        songs = songs['tracks']

    results = []

    for song in songs:

        try:
            track_id = song['id']
            track_name = song['name']
            artists = [element['name'] for element in song['artists']]
            popularity = song['popularity']
            preview = song['preview_url']
            image = format(song['album']['images'][0]['url'], '')

            results.append([track_id, track_name, artists, popularity, preview, image])

        except:
            pass

    #Este es el df de las canciones resultado de la búsqueda del usuario        
    df_user_search = pd.DataFrame(results, columns = ['track_id', 'track_name', 'track_artists', 'track_popularity', 'preview', 'image'])

    songs_to_query = df_user_search['track_id'].to_list()

    user_pick_songs_audio_features = []

    # Se hace una petición de los audio features de la query del usuario
    user_pick_features_request = spotify.audio_features(songs_to_query)

    for song in user_pick_features_request:

        try:
            track_id = song['id']
            danceability = song['danceability']
            energy = song['energy']
            key = song['key']
            loudness = song['loudness']
            mode = song['mode']
            speechiness = song['speechiness']
            acousticness = song['acousticness']
            instrumentalness = format(song['instrumentalness'], '.10f')
            liveness = song['liveness']
            valence = song['valence']
            tempo = song['tempo']
            duration_ms = song['duration_ms']
            time_signature = song['time_signature']

            user_pick_songs_audio_features.append([track_id, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, time_signature])

        except:
            pass

    #Este es el df con los audio features de las canciones buscadas anteriormente
    df_user_search_audio_features = pd.DataFrame(user_pick_songs_audio_features, columns = ['track_id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'])

    df_user_pick = df_user_search.copy()
    df_user_pick = df_user_pick.merge(df_user_search_audio_features, how = 'inner', left_on = 'track_id', right_on = 'track_id') #Aquí se combinan ambos dfs mediante un merge por el track_id
    df_user_pick = df_user_pick.sort_values(by = 'track_popularity', ascending = False).reset_index(drop = True)
    df_user_pick.index = df_user_pick.index + 1

    return df_user_pick



def songs_clustering(df_songs_to_recomend, df_user_pick, audio_features = ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode','speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']):
    
    song_id = df_user_pick['track_id'].to_list()
    
    user_pick = df_user_pick.copy()
        
    # Nos aseguramos de que las canciones que ha elegido el usuario no está en las canciones donde se hará el clustering.
    if len(song_id) > 1:
        songs_to_recomend = df_songs_to_recomend[~df_songs_to_recomend['track_id'].isin(song_id)].copy()
    else:
        songs_to_recomend = df_songs_to_recomend[df_songs_to_recomend['track_id'] != song_id[0]].copy() 
        
    X = songs_to_recomend[audio_features]
    
    # Normalización de audio features
    x_scaler = MinMaxScaler()
    X = x_scaler.fit_transform(X)

    # Inicializar un una lista de listas vacía para almacenar los resultados de nuestro "método del codo" de epsilon para el DBSCAN.
    epsilon_results = []
    # Iterar sobre diferentes valores de epsilon para determinar qué modelo se adapta mejor al conjunto de datos resultante de las búsquedas del usuario.
    for i in stqdm(np.arange(0.1, 1, 0.005), desc=f'Spytifying the music :smile:'):
        dbscan = DBSCAN(eps=i, metric='euclidean')
        dbscan.fit(X)

        # Contar el número de clusters y outliers
        cluster_count = Counter(dbscan.labels_)
        num_clusters = len(cluster_count) - (1 if -1 in cluster_count else 0)  # Excluir el ruido/outliers (-1)
        num_outliers = cluster_count[-1] if -1 in cluster_count else 0

        sil_score = 0 if num_clusters <= 1 else silhouette_score(X, dbscan.labels_)
        
        epsilon_results.append([i, num_clusters, num_outliers, sil_score])
    
    # Almacenar los resultados en el DataFrame
    epsilon = pd.DataFrame(epsilon_results, columns=['Epsilon', 'NumClusters', 'NumOutliers', 'SilhouetteScore'])
            
    #Lanzamos el modelo de clustering nuevamente con el mejor epsilon que cumpla las condiciones de 1. Más de 2 clusters, 2. Con mayor SilohouetteScore y 3. Con menor cantidad de elementos outliers/ruido
    #Si no tiene outliers que se quede con la mejor configuración aunque sean 2 clusters
    best_eps = epsilon[epsilon['NumClusters'] >= 2].sort_values(by = ['SilhouetteScore', 'NumOutliers'], ascending = [False, True]).iloc[0,0]

    dbscan = DBSCAN(eps = best_eps, metric = 'euclidean')
    dbscan.fit(X)
    
    #Se agregan las etiquetas de los clusters a las canciones a recomendar
    songs_to_recomend['Label'] = dbscan.labels_

    #Aquí nos aseguramos de que todas las features con las que se hace el cluster son numéricas y así no falle la agrupación para crear los centroides del cluster
    for feature in audio_features:
        songs_to_recomend[feature] = pd.to_numeric(songs_to_recomend[feature], errors='coerce')

    #Se calculan los centroides de cada cluster (la media de todos los valores de los audio features por cada grupo/etiqueta)
    df_centroides = songs_to_recomend[songs_to_recomend['Label'] >= 0].groupby("Label", as_index = False).agg({feature : "mean" for feature in audio_features})

    #Se calcula la distancia entre los picks del usuario y los centroides resultado del clustering, y se etiqueta o se le da la label del cluster más cercanos según los valores de los audio features
    user_pick['Label'] = user_pick.apply(lambda row : user_pick_labeling(row, df_centroides, audio_features), axis = 1)
    
    #Etiqueta del centroide más cercano a los picks del usuario
    centroid_label = user_pick['Label'].value_counts().sort_values(ascending = False).index[0]
    
    return user_pick, songs_to_recomend, centroid_label, epsilon


def songs_to_playlist(user_pick, songs_to_recomend, combine, songs_number, shuffle):
    
    # Crea una lista de track_ids de los picks del usuario, en caso de fallo crea una lista vacía.
    try:
        user_pick_id = user_pick['track_id'].to_list()
    except:
        user_pick_id = []
    
    # Crea una lista de track_ids de las canciones que se van a recomendar al usuario.
    songs_to_recomend_ids = songs_to_recomend['track_id'].to_list()
    songs_to_playlist = user_pick_id.copy()
    
    """Se crea una lista con los picks del usuario y las canciones recomendadas de tantos elementos
       como el usuario quiera (songs_number) solo si el usuario acepta que se combinen, de lo contrario,
       solo se guardan las canciones recomendadas.
    """
    if combine and len(songs_to_playlist) < songs_number:
        songs_to_add = songs_to_recomend_ids[:songs_number - len(songs_to_playlist)]
        songs_to_playlist.extend(songs_to_add)

    else:
        songs_to_add = songs_to_recomend_ids[:songs_number + 1]
        songs_to_playlist.extend(songs_to_add)
        
    # Si el usuario lo desea, se ordenan de manera aleatoria, de lo contrario se ordenarán según las preferencias del usuario por popularidad.
    if shuffle:
        random.shuffle(songs_to_playlist)
        
    return songs_to_playlist


def post_playlist(spotify,
                  REDIRECT_URI,
                  songs_to_playlist, 
                  name = '', 
                  description = None):
    
    if name == '' or name == None:

        name = 'Super cool playlist' 
        description = 'Your favorite playlist for'
        
        datetime_now = datetime.now()
        
        name = f'{name} {datetime.strftime(datetime_now, "%b%Y")}'
        description = f'{description} {datetime.strftime(datetime_now, "%Y")}'

    description = f'{description} - playlist created using Spytify🐍 {REDIRECT_URI.split("https://")[-1]}'

    user = spotify.me()['id']
    
    new_playlist = spotify.user_playlist_create(user = user, name = name, public = True, collaborative = False, description = description)

    playlist_id = new_playlist['id']
    playlist_url = new_playlist['href']

    batches = list(batch_generator(songs_to_playlist))

    for batch in stqdm(batches, desc='Creating your playlist...'):
        spotify.user_playlist_add_tracks(user, playlist_id, batch)

    return playlist_url, name, len(songs_to_playlist)


def get_all_playlist_items(spotify, playlist_id):
    
    all_items = []

    # Primera solicitud para obtener la primera página
    playlist_items = spotify.playlist_tracks(playlist_id)

    # Agregar los elementos de la primera página a la lista general
    all_items.extend(playlist_items["items"])

    # Obtener las páginas adicionales utilizando next()
    while playlist_items["next"]:
        playlist_items = spotify.next(playlist_items)
        all_items.extend(playlist_items["items"])

    return all_items

def user_playlist_stats_request(spotify, playlist_url):
    
    # Extrae el playlist_id del input del usuario
    if '?' in playlist_url:
        playlist_id = playlist_url.split('/')[-1].split('?')[0]
    else:
        playlist_id = playlist_url.split('/')[-1]

    # Hace una varias peticiones a la API de Spotify para extraer todas las canciones/items de un playlist
    playlists = get_all_playlist_items(spotify, playlist_id)

    results = []

    for track in stqdm(playlists, desc = "Getting your playlist's tracks"):

        track_id = track['track']['id']
        track_name = track['track']['name']
        track_artists = [artist['name'] for artist in track['track']['artists']]
        track_popularity = track['track']['popularity']

        results.append([track_id, track_name, track_artists, track_popularity])

    df = pd.DataFrame(results, columns = ['track_id', 'track_name', 'track_artists', 'track_popularity'])
    df = df.drop_duplicates(['track_id'])
    df = df.dropna(axis = 0)
    songs_list = df['track_id'].to_list()

    songs_audio_features = []

    # Se crean batches de 100 elementos para luego pedir los audio features via API
    batches = list(batch_generator(songs_list))

    for batch in stqdm(batches,desc="Getting audio features..."):

        # Se solicitanlos audio features de las canciones de cada batch
        features_request = spotify.audio_features(batch)

        for song in features_request:

            try:
                track_id = song['id']
                danceability = song['danceability']
                energy = song['energy']
                key = song['key']
                loudness = song['loudness']
                mode = song['mode']
                speechiness = song['speechiness']
                acousticness = song['acousticness']
                instrumentalness = format(song['instrumentalness'], '.10f')
                liveness = song['liveness']
                valence = song['valence']
                tempo = song['tempo']
                duration_ms = song['duration_ms']
                time_signature = song['time_signature']

                songs_audio_features.append([track_id, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, time_signature])

            except:
                pass
        sleep(0.5)

    df_audio_features = pd.DataFrame(songs_audio_features, columns = ['track_id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'])
    df_merge = df.copy()
    df_merge = df_merge.merge(df_audio_features, how = 'inner', left_on = 'track_id', right_on = 'track_id') # Se crea un dataframe con las canciones de la playlist que quiere el usuario más sus audio features.

    return df_merge

def get_songs_to_playlist(spotify, items, query_type):

    """Esta función permite a partir de una lista de items similares (artistas o canciones), 
       hacer peticiones a la API de Spotify para conseguir las canciones para crear un playlist."""
    
    results = []

    if query_type == 'artist':

        for artist in stqdm(items, desc="Getting your favorite artist's songs"):
            artist_id = spotify.search(q = artist, limit = 1, offset = 0, type = query_type, market = 'ES')
            artist_id = artist_id['artists']['items'][0]['id']
            songs = spotify.artist_top_tracks(artist_id = artist_id, country = 'ES')
            songs = songs['tracks']

            for song in songs:
                try:
                    track_id = song['id']
                    track_name = song['name']
                    artists = [element['name'] for element in song['artists']]
                    popularity = song['popularity']
                    preview = song['preview_url']
                    image = format(song['album']['images'][0]['url'], '')

                    results.append([track_id, track_name, artists, popularity, preview, image])

                except:
                    pass

    else:
        for song in stqdm(items, desc="Getting your favorite songs"):
            song = spotify.search(q = song, limit = 1, offset = 0, type = query_type, market = 'ES')
            song = song['tracks']['items'][0]
            try:
                track_id = song['id']
                track_name = song['name']
                artists = [element['name'] for element in song['artists']]
                popularity = song['popularity']
                preview = song['preview_url']
                image = format(song['album']['images'][0]['url'], '')

                results.append([track_id, track_name, artists, popularity, preview, image])

            except:
                pass

    df = pd.DataFrame(results, columns = ['track_id', 'track_name', 'artists', 'track_popularity', 'preview', 'image'])

    return df, len(df)

def success_playlist(playlist_url, playlist_name, number_songs_uploaded):
    # Mensaje de éxito
    success_popup = ['Your playlist', f'has been created and {number_songs_uploaded} songs were added to it']
    playlist_url = playlist_url.replace('//api.', '//open.').replace('/v1/playlists/', '/playlist/')
    # Crear el mensaje con st.markdown
    message = f"{success_popup[0]} [{playlist_name}]({playlist_url}) {success_popup[1]}"
    st.success(message)
    st.balloons()

def get_music_genres(force = False):

    """Esta función abre una lista de géneros musicales desde un fichero de sources, 
    y si no lo consigue hace un scrappeo web estático de una fuente que tiene más de 5000 géneros musicales"""

    file_name = "sources/genres_list.pkl"

    if not(force) and os.path.exists(file_name):
        # Carga el archivo pickle si existe
        with open(file_name, 'rb') as file:
            return pickle.load(file)

    genres_list = []

    url = "https://www.everynoise.com/everynoise1d.html" #page with genres ordered by popularity
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        tds = soup.find_all("td", class_ = "note")

        for td in tds:
            if not(td.text.isdigit()):
                genres_list.append(td.text)

    with open(file_name, 'wb') as file:
        pickle.dump(genres_list, file)

    return genres_list

def users_pick_html(df_user_pick):

    #Esta función crea una tabla con previews de audio e imágenes de los albumes de las canciones que ha solicitado el usuario.
    table_html = '<table><tr><th>Song Number</th><th>Track Name</th><th>Artists</th><th>Track Popularity</th><th>Audio Preview</th><th>Album Image</th></tr>'
    for idx, song in df_user_pick.iterrows():
        audio_object = f'<audio controls src="{song["preview"]}" type="audio/mp3"></audio>'
        song_image = f'<img src="{song["image"]}" width="100" height="100">'  # Ajusta el tamaño según tus necesidades
        artists_html = ''.join([f'<li>{artist}</li>' for artist in song['track_artists']])
        artists_html = f'<ul>{artists_html}</ul>'
        table_html += f'<tr><td>{idx}</td><td>{song["track_name"]}</td><td>{artists_html}</td><td>{song["track_popularity"]}</td><td>{audio_object}</td><td>{song_image}</td></tr>'
    table_html += '</table>'

    return table_html

def highlight_value(column): #Esta función resalta los valores máximos de las columnas de un dataframe
    is_max = column == column.max()
    return ['background-color: #2ECC71' if value else '' for value in is_max]