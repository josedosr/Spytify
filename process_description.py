import streamlit as st
from stqdm import stqdm
import time
from time import sleep

#Librerías de dataframes y arrays
import numpy as np
import pandas as pd

# WordCloud
from workcloud import WordCloud
from PIL import Image

#Librerías para interactuar con Spotify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import MemoryCacheHandler
from bokeh.models.widgets import Div

#Importar las funciones de los procesos
#from funciones import get_all_playlist_items, user_playlist_stats_request, batch_generator
from eda_funciones import eda_ui, eda_playlist_query, popularidad_artista, popularidad_genero, genero_count, histograma_duration_ms, popularidad_playlist, popularidad_fecha_genero, line_polar_playlist, top_artists_plot
from funciones import get_music_genres, users_pick_html, success_playlist, highlight_value

def process_description():

    if 'df_recommendations_cache' in st.session_state:
        df_recommendations_cache = st.session_state['df_recommendations_cache']

    if 'df_user_query' in st.session_state:
        df_user_query = st.session_state['df_user_query']

    if 'epsilon' in st.session_state:
        epsilon = st.session_state['epsilon']

    show_success = False
    show_cluster = False
    show_playlist = False

    st.title('Spytify Process Description')

    st.markdown(f"""In this section we explain you how to use the Main App section step by step and how the app :computer: works internally :robot_face:.
                    \nStarting from `User Input` submitting your queries, 
                    then making your input song choices to start the `Clustering Recommendations` process, 
                    proceeding through `Exploratory Data Analysis` to visualize important patterns in your data, 
                    and finally creating the ultimate set of music in the `Playlist Export` section.""")
    
    st.markdown('## User Input')

    st.markdown("""#### Step 1: Enter Preferences

Complete the form below with your musical preferences:

- *Query:* Enter the name of an artist or a song that interests you.
- *Query Results:* Enter how many results you want in your Track query, if you make an Artist query the results will be their top 10 songs.
- *Musical Genres:* Select up to 6 genres you like, such as Pop, Rock, Electronic, Rap, R&B, or Latin music.
""")
    #User input form example
    with st.form('User_Input'):
            query_type_test = st.radio(label='Select your query type',
                        options=('Track', 'Artist'),
                        index=0,
                        disabled=False,
                        horizontal=True,
                        )
            
            query_text_test = st.text_input(label= 'Write your query',
                        max_chars=100,
                        value='Outside')
            
            limit_test = st.number_input(label='Enter how many results you want on your track query (1-50). If your query type is an artist please ignore this field',
                        min_value=1,
                        max_value=50,
                        value=20,
                        step=1)

            genre_test = st.multiselect('Select the genre(s) that you want to lookup data',
                        options = ['edm', 'rap', 'pop', 'latin', 'rock', 'r&b'],
                        default = ['edm', 'rap', 'pop', 'latin', 'rock', 'r&b'],
                        key='genres_multiselect_',
                        max_selections=10)

            submitted_test = st.form_submit_button("Submit")

            if submitted_test:
                st.info(f'We are processing your {query_type_test} requests, please wait. This may take a while...')
                show_success = True

    st.markdown("""#### Step 2: Processing Preferences

Once you submit your preferences, the app will process the information and make calls to the Spotify API using the Spotipy library. This will generate two sets of data:

- **Found Songs:** A list of songs related to your search.
- **Recommended Playlists + Songs:** A list of playlists and songs based on your selected musical genres.
""")
    
    if show_success:
        st.success('Your request was succesfully done')

    st.markdown("""#### Step 3: Results

Explore the results of your search:
##### Found Songs

Visualize the songs related to your search, so you can discover new artists or find your favorite songs.""")

    if show_success:
        with st.expander(label="DataFrame", expanded=False):
            st.dataframe(df_user_query[['track_name', 'track_artists', 'track_popularity']], width = 1000)

    st.markdown("""##### Recommended Playlists and Songs

Discover personalized playlists that match your favorite musical genres. Explore new songs and expand your musical repertoire.
""")
    if show_success:
        with st.expander(label="DataFrame", expanded=False):
            st.dataframe(df_recommendations_cache[['track_name', 
                                                    'track_artists', 
                                                    'track_popularity', 
                                                    'track_album_name', 
                                                    'track_album_release_date',
                                                    'playlist_name', 
                                                    'playlist_genre']])
        # Generate wordcloud texts
        text = " ".join(df_recommendations_cache.explode('track_artists', ignore_index=True)['track_artists'])

        # Transparent Mask
        mask = np.array(Image.new("RGB", (800, 400), (255, 255, 255)))  # Fondo blanco
        mask[:, :, 2] = 0  # Establecer canal alfa en 0 para transparencia

        # Apply mask
        wordcloud = WordCloud(width=2400, height=1200, mode="RGBA", background_color=None, colormap='viridis', mask=mask).generate(text)
        st.image(wordcloud.to_array(), width=1200)

    st.markdown("""After entering your preferences and obtaining Spotify datasets, we move on to the exciting step of launching the clustering model with DBSCAN.
                \n## Clustering Recommendations""")

    st.markdown("""#### Step 4: Choose Songs and Audio Features

Select the songs resulting from your initial search. You have a table with the album photos and audio previews to facilitate your choices. Additionally, choose the audio features you want to incorporate into the model.""")
    
    
    #Se crea una tabla formato HMTL con información de las canciones del usuario, con preview de audio e imagen del album disponible        
    table_html = users_pick_html(df_user_query)

    with st.expander(label="User Query Table", expanded=False):
        #Mostrar la tabla HTML personalizada
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("")

    songs_options = [f'{idx}. {song}' for idx, song in zip(df_user_query.index, df_user_query['track_name'])]
    audio_features = ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode','speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
            
    with st.form('Index_list'):
            
        chosen_songs_test = st.multiselect(label="Select the songs that you want to make the clustering process", 
                                    options=songs_options, 
                                    default=songs_options[:5])

        audio_features_to_cluster_test = st.multiselect(label="Select the audio features you want to use in the songs analysis", 
                                                    options=audio_features, 
                                                    default=audio_features)
        
        st.markdown('''### Audio Features Descriptions

- **Danceability**: indicates how suitable a track is for dancing (values range from 0.0 - 1.0).
- **Energy**: represents the perceptual intensity and activity of a track (values range from 0.0 - 1.0).
- **Key:** estimated overall key of the track mapped to pitches using standards (values range -1 - 11).
- **Loudness:** measures the overall loudness of a track in decibels (dB) (values range -60- 0 dB).
- **Mode:** indicates the modality (major or minor) of a track.
- **Speechiness:** detects the presence of spoken words in a track.
- **Acousticness:** confidence measure from 0.0 - 1.0 indicating whether the track is acoustic.
- **Instrumentalness:** predicts whether a track contains no vocals.
- **Liveness:** detects the presence of an audience in the recording.
- **Valence:** describes the musical positiveness conveyed by a track (values range 0.0 - 1.0).
- **Tempo:** overall estimated tempo of a track in beats per minute (BPM).
- **Duration:** duration of the song in milliseconds.

If you want to review more information about the audio features review the [Spotify API Documentation - Audio Features](https://developer.spotify.com/documentation/web-api/reference/get-audio-features)''')

        submit_cluster = st.form_submit_button("Submit")

        if submit_cluster:
            st.info('The clustering process had started. This may take a while...')
            show_cluster = True

    st.markdown("""#### Step 5: DBSCAN Configuration

Upon submitting the chosen songs and audio features, DBSCAN initiates the iteration process over different epsilon values to determine the optimal configuration for the model. This algorithm clusters songs based on their local density, resulting in more accurate and personalized outcomes.""")

    if show_cluster:
        for i in stqdm(range(180), desc=f'Spytifying the music :smile:'):
            sleep(0.05)
        st.success(f"The clustering process is completed, now you can review your election's group/label.")
                
    st.markdown("""#### Step 6: Evaluation with Silhouette Score and Customized Results

After launching the model, you'll obtain a DataFrame with detailed information on the best model, including epsilon, the number of outliers, the number of groups, and the Silhouette Score. The Silhouette Score evaluates the coherence and separation of the clusters, helping you select the most robust grouping. Additionally, you'll have the assignment of the most common group among your choices.

This process offers dynamic and personalized grouping tailored to your musical preferences.""")
    
    if show_cluster:
        st.markdown('##### Clustered Results')
        with st.expander(label="Clustered Results Dataframe", expanded=False):
            st.dataframe(df_user_query[['track_name', 'track_artists', 'track_popularity', 'Label']], width = 1000)

        st.markdown('##### Best Model')
        with st.expander(label="Best Model Dataframe", expanded=False):
            df_epsilon = epsilon[epsilon['NumClusters'] >= 2].sort_values(by = ['SilhouetteScore', 'NumOutliers'], ascending = [False, True]) #Dataframe ordenado para mostrar el mejor modelo
            df_epsilon = df_epsilon.style.apply(highlight_value, subset = ['SilhouetteScore'])
            st.dataframe(df_epsilon, width = 600)


    st.markdown("""## Exploratory Data Analysis

This section showcases visualizations created using the Plotly package to analyze playlists and songs data.

1. **Artist Popularity Average (Bar Chart):**
   - Displays the average popularity of musical artists.
   - *X-axis:* Artist names, *Y-axis:* Average popularity.""")
    
    st.plotly_chart(popularidad_artista(df_recommendations_cache), use_container_width= True)

    st.markdown("""2. **Average Popularity by Genre (Pie Chart):**
   - Illustrates the distribution of average popularity across genres.
   - Data grouped by playlist genre, showcasing average popularity for the top 20 genres.""")
    
    st.plotly_chart(popularidad_genero(df_recommendations_cache), use_container_width= True)
    
    st.markdown("""3. **Songs per Genre Distribution (Pie Chart):**
   - Shows the distribution of songs per genre in a playlist.
   - Top 20 genres with the highest song count are highlighted.""")
    
    st.plotly_chart(genero_count(df_recommendations_cache), use_container_width= True)
    
    st.markdown("""4. **Popularity Timeline per Genre (Area Chart):**
   - Visualizes the trend of average song popularity over time for different genres.
   - *X-axis:* Year, *Y-axis:* Average song popularity, *Coloured areas:* Genres.""")
            
    st.plotly_chart(popularidad_fecha_genero(df_recommendations_cache), use_container_width= True)

    st.markdown("""5. **Popularity per Playlist (Bar Chart):**
   - Displays the average popularity of music playlists.
   - *X-axis:* Playlist names, *Y-axis:* Average popularity.""")
    
    st.plotly_chart(popularidad_playlist(df_recommendations_cache), use_container_width= True)
    
    st.markdown("""6. **Track Length per Genre (Histogram):**
   - Depicts the distribution of song durations for each genre.
   - *X-axis:* Song duration in minutes, *Y-axis:* Frequency of songs.""")
    
    st.plotly_chart(histograma_duration_ms(df_recommendations_cache), use_container_width= True)
    
    st.markdown("""7. **Top 10 Most Repeated Artists (Bar Chart):**
   - Highlights the most frequently occurring artists in the playlist.
   - Counts occurrences, presents top ten artists.""")
    
    st.plotly_chart(top_artists_plot(df_recommendations_cache), use_container_width= True)
    
    st.markdown("""8. **Audio Features Average (Polar Chart):**
   - Shows the average music characteristics in a specific playlist.
   - Characteristics represented as radial axes, average values determine radial length.""")
    
    playlists = df_recommendations_cache["playlist_name"].unique()
    audio_features = ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode','speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    
    col1, col2 = st.columns([2, 8])
    playlist_names = col1.multiselect(label="Select playlists you want to analyze", 
                                                options = playlists, 
                                                default = playlists[:3])

    audio_features_graph = col1.multiselect(label="Select the audio features you want to use in the playlist analysis", 
                                                options=audio_features, 
                                                default=audio_features)
    
    col2.plotly_chart(line_polar_playlist(df_recommendations_cache, playlist_names, audio_features_graph), use_container_width= True)
    
    st.markdown('Each visualization provides valuable insights into different aspects of the playlist, enhancing understanding and exploration.')


    st.markdown("""This process allows you to explore and discover new songs in a personalized way.
                
## Playlist Export
                
After the clustering process, users have the option to create a personalized playlist by combining their selected songs with those from the majority group in the clustering results.

#### Step 7: Playlist Customization:

Users fill out a form where they name their playlist, provide an optional description, choose between most or least popular songs, decide whether to include their selected songs, specify the number of songs, and opt for random or popularity-based order.""")
    
    with st.form('Playlist_export'):

        name_test = st.text_input(label= 'Name your playlist',
                    max_chars=100,
                    value='Super cool playlist')
        
        description_test = st.text_input(label= 'Add a description to your playlist ',
                    max_chars=100,
                    value='Your favorite playlist for 2024')

        combine_option_test = st.radio(label='Do you want to include your picks in the playlist?',
                    options=('Yes', 'No'),
                    index=0,
                    disabled=False,
                    horizontal=True,
                    )
        
        popularity_option_test = st.radio(label='Which recommended songs do you want to include in your playlist?',
                    options=('Most popular', 'Less popular'),
                    index=0,
                    disabled=False,
                    horizontal=True,
                    )
                    
        shuffle_option_test = st.radio(label='Do you want to order your playlist songs randomly? Otherwise the most/less popular songs will be picked',
                    options=('Yes', 'No'),
                    index=0,
                    disabled=False,
                    horizontal=True,
                    )
        
        songs_number = st.slider(label='How many songs do you want to add to your playlist?',
                                min_value = 1,
                                max_value = 1000,
                                value=100)
                        
        playlist_submit_test = st.form_submit_button("Submit")

        if playlist_submit_test:
            playlist_url = 'https://open.spotify.com/playlist/1r2eUDfANNje9kXY8eBA72'
            playlist_name = 'Super cool playlist'
            number_songs_uploaded = 100
            show_playlist = True

    st.markdown("""#### Step 8: Playlist Pre-processing:

Upon form submission, an algorithm processes the input and configures a playlist in list format, containing the IDs of the selected songs and those from the majority group in the clustering results.""")
    
    if show_playlist:
        st.info('Your playlist is ready to be posted')

    st.markdown("""#### Step 9: Playlist post via API Calls with Spotipy:
                
Using Spotipy, calls to the Spotify API are made to:
    - Create the playlist with the given name and description.
    - Add songs to the playlist in batches of 100, making requests to the Spotify API to gradually populate the playlist.

This seamless process ensures that users can easily curate their personalized playlists based on their preferences and the clustering results.""")
    
    if show_playlist:
        for i in stqdm(range(10), desc='Creating your playlist...'):
            sleep(0.5)

        st.markdown("""#### Step 10: Enjoy your new playlist :tada::
                    
Now you can listen to your personalized playlist created using DBSCAN clustering on Spotify.

Don't forget to save and share your playlist with friends for a collective musical experience!
""")      
        success_playlist(playlist_url, playlist_name, number_songs_uploaded)

if __name__ == "__process_description__":
    process_description()
