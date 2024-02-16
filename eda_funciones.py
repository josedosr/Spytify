import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
    
# Definir funciones para análisis y visualización

def popularidad_artista(df, n_artistas=20):

    df_pop_art = df.copy()

    #df_pop_art["track_artists"] = df_pop_art["track_artists"].apply(lambda x : eval(x) if pd.notna(x) else [])
    df_pop_art = df_pop_art.explode("track_artists", ignore_index=True)
    df_pop_art = df_pop_art.groupby(by = "track_artists", as_index = False).agg({"track_popularity" : "mean"})\
                   .sort_values("track_popularity", ascending = False)\
                   .reset_index(drop = True)

    fig = px.bar(data_frame = df_pop_art.iloc[:n_artistas],
                 x = "track_artists",
                 y = "track_popularity",
                 color_discrete_sequence=px.colors.qualitative.Dark2)
    
    fig.update_layout(title="Artist popularity average",
                      xaxis_title="Artist",
                      yaxis_title="Popularity average")

    fig.update_xaxes(title_font=dict(size=18, family='Arial', color='#e5383b'),
                     tickfont=dict(size=14, family='Arial', color='#939393'))
    fig.update_yaxes(title_font=dict(size=18, family='Arial', color='#e5383b'),
                     tickfont=dict(size=14, family='Arial', color='#939393'))
    
    return fig

def popularidad_genero(df, n_generos=20):
    df = df.copy()
    
    df_pop_gen = df.groupby(by = "playlist_genre", as_index = False).agg({"track_popularity" : "mean"})\
               .sort_values("track_popularity", ascending = False)\
               .reset_index(drop = True)

    fig = px.pie(data_frame = df_pop_gen.iloc[:n_generos],
                 names = "playlist_genre",
                 values = "track_popularity",
                 color_discrete_sequence=px.colors.qualitative.Plotly)

    fig.update_layout(title="Average popularity by genre",
                      legend_title="Genre",
                      legend=dict(title_font=dict(size=18, family='Arial', color='#e5383b'),
                                  font=dict(size=14, color='#939393')))
    
    return fig

def genero_count(df, n_generos=20):
    df_genero_count = df.groupby(by = ["playlist_genre"], as_index = False)\
                        .agg({"track_id" : "count"})\
                        .sort_values("track_id", ascending = False)\
                        .reset_index(drop = True)
    
    fig = px.pie(data_frame = df_genero_count.iloc[:n_generos],
                 names = "playlist_genre",
                 values = "track_id",
                 color_discrete_sequence=px.colors.qualitative.Plotly)

    fig.update_layout(title="Songs obtained per genre distribution",
                      legend_title="Genre",
                      legend=dict(title_font=dict(size=18, family='Arial', color='#e5383b'),
                                  font=dict(size=14, color='#939393')))
    
    return fig

def popularidad_fecha_genero(df):

    df["year"] = df["track_album_release_date"].apply(lambda x: int(x[:4]))

    # Filtra el DF para incluir solo las filas con años mayores a 1800
    df = df[df["year"] > 1800]

    df_funcion = df.groupby(["year", "playlist_genre"], as_index = False).agg({"track_popularity" : "mean"})
    
    fig = px.area(data_frame = df_funcion,
                 x = "year",
                 y = "track_popularity",
                 color = "playlist_genre",
                 color_discrete_sequence=px.colors.qualitative.Plotly)

    fig.update_layout(title="Popularity timeline per genre",
                      xaxis_title="Year",
                      yaxis_title="Popularity average")

    fig.update_xaxes(title_font=dict(size=18, family='Arial', color='#e5383b'),
                     tickfont=dict(size=14, family='Arial', color='#939393'))
    fig.update_yaxes(title_font=dict(size=18, family='Arial', color='#e5383b'),
                     tickfont=dict(size=14, family='Arial', color='#939393'))

    return fig

def popularidad_playlist(df, n_playlist=20):
    """Media de popularidad por playlist"""
    
    df_funcion = df.groupby("playlist_name", as_index = False)\
                   .agg({"track_popularity" : "mean"})\
                   .sort_values("track_popularity", ascending = False)\
                   .reset_index(drop = True)
    
    fig = px.bar(data_frame = df_funcion.iloc[:n_playlist],
                 x = "playlist_name",
                 y = "track_popularity",
                 color_discrete_sequence=px.colors.qualitative.Set1)

    fig.update_layout(title="Popularity per playlist",
                      xaxis_title="Playlist",
                      yaxis_title="Popularidad average")

    fig.update_traces(marker_color='rgb(105,250,225)', marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)

    fig.update_layout(bargap=0.1)

    fig.update_xaxes(title_font=dict(size=14, family='Arial', color='#e5383b'),
                     tickfont=dict(size=9.5, family='Arial', color='#939393'))
    fig.update_yaxes(title_font=dict(size=14, family='Arial', color='#e5383b'),
                     tickfont=dict(size=9.5, family='Arial', color='#939393'))

    return fig

def histograma_duration_ms(df):

    df["duration_ms"] = df["duration_ms"]/(1000*60)

    fig = px.histogram(data_frame = df,
                       x = "duration_ms",
                       marginal = "box",
                       color = "playlist_genre",
                       color_discrete_sequence=px.colors.qualitative.Plotly) #aquí podemos agregar nbins = 20 si queremos

    fig.update_layout(title="Track length per genre",
                      xaxis_title="Track length (minutes)",
                      yaxis_title="Frequency")

    fig.update_xaxes(title_font=dict(size=14, family='Arial', color='#e5383b'),
                     tickfont=dict(size=9.5, family='Arial', color='#939393'))
    fig.update_yaxes(title_font=dict(size=14, family='Arial', color='#e5383b'),
                     tickfont=dict(size=9.5, family='Arial', color='#939393'))
    
    return fig

def top_artists_plot(df, top_n=10):
    """Generar un gráfico de los artistas más repetidos en la playlist"""

    # Explode la columna 'track_artists' para manejar listas en celdas
    df_expanded = df.explode("track_artists", ignore_index=True)

    # Obtener los artistas más repetidos
    top_artists = df_expanded["track_artists"].value_counts().head(top_n)

    # Crear un DataFrame para el gráfico
    data = pd.DataFrame({
        "Artist": top_artists.index,
        "Count": top_artists.values
    })

    # Crear el gráfico de barras
    fig = px.bar(data_frame=data,
                 x="Artist",
                 y="Count",
                 labels={"Count": "Tracks"},
                 title=f"Top {top_n} most repeated artists in the playlist",
                 height=500,
                 color_discrete_sequence=px.colors.qualitative.Dark2)

    fig.update_xaxes(title_font=dict(size=18, family='Arial', color='#e5383b'),
                     tickfont=dict(size=14, family='Arial', color='#939393'))
    fig.update_yaxes(title_font=dict(size=18, family='Arial', color='#e5383b'),
                     tickfont=dict(size=14, family='Arial', color='#939393'))

    return fig

def line_polar_playlist(df, playlist_names, audio_features):           
    
    scaler = MinMaxScaler()
    df_normalized = df[['playlist_name'] + audio_features]
    df_normalized.iloc[:, 1:] = scaler.fit_transform(df_normalized.iloc[:, 1:])

    fig = px.line_polar(title="Playlist Audio Features Average")

    for i, playlist_name in enumerate(playlist_names):
        df_polar = df_normalized[df_normalized["playlist_name"] == playlist_name]

        df_polar = df_polar.groupby("playlist_name", as_index=False).agg({col: "mean" for col in df_polar.columns[1:]})

        # Seleccionar un color de las paletas de Plotly
        color = px.colors.qualitative.Dark2[i % len(px.colors.qualitative.Dark2)]

        fig.add_trace(px.line_polar(data_frame=df_polar.iloc[:, 1:],
                                     r=df_polar.iloc[:, 1:].values[0],
                                     theta=df_polar.columns[1:],
                                     line_close=True,
                                     color_discrete_sequence=[color]).update_traces(name=playlist_name, showlegend=True).data[0])

    fig.update_layout(polar=dict(radialaxis=dict(title='Average',
                                                title_font_color='black', 
                                                tickfont_color='black', 
                                                categoryarray=df_polar.columns[1:])), legend_title='Playlists')

    return fig

def line_polar_artists(df, artists_names, audio_features):           
    
    scaler = MinMaxScaler()
    df_normalized = df[['track_artists'] + audio_features]
    df_normalized.iloc[:, 1:] = scaler.fit_transform(df_normalized.iloc[:, 1:])
    df_normalized = df_normalized.explode("track_artists", ignore_index=True)

    fig = px.line_polar(title='Artist Audio Features Average')

    for i, artists_name in enumerate(artists_names):
        df_polar = df_normalized[df_normalized['track_artists'] == artists_name]

        df_polar = df_polar.groupby('track_artists', as_index=False).agg({col: 'mean' for col in df_polar.columns[1:]})

        # Seleccionar un color de las paletas de Plotly
        color = px.colors.qualitative.Dark2[i % len(px.colors.qualitative.Dark2)]

        fig.add_trace(px.line_polar(data_frame=df_polar.iloc[:, 1:],
                                     r=df_polar.iloc[:, 1:].values[0],
                                     theta=df_polar.columns[1:],
                                     line_close=True,
                                     color_discrete_sequence=[color]).update_traces(name=artists_name, showlegend=True).data[0])

    fig.update_layout(polar=dict(radialaxis=dict(title='Average',
                                                title_font_color='black', 
                                                tickfont_color='black', 
                                                categoryarray=df_polar.columns[1:])), legend_title='Artists')

    return fig


# Agregar Streamlit UI
def eda_ui(df):

    df_copy = df.copy()

    st.plotly_chart(popularidad_artista(df_copy), use_container_width= True)
            
    col1, col2 = st.columns([1, 1])
    col1.plotly_chart(popularidad_genero(df_copy), use_container_width= True)
    col2.plotly_chart(genero_count(df_copy), use_container_width= True)
    st.plotly_chart(popularidad_fecha_genero(df_copy), use_container_width= True)

    col1, col2 = st.columns([1, 1])
    col1.plotly_chart(popularidad_playlist(df_copy), use_container_width= True)
    col2.plotly_chart(histograma_duration_ms(df_copy), use_container_width= True)
    st.plotly_chart(top_artists_plot(df_copy), use_container_width= True)
    
    col1, col2 = st.columns([2, 8])
    playlists = df_copy["playlist_name"].unique()
    #playlist_names = col1.selectbox("Selecciona una playlist", songs_to_recomend["playlist_name"].unique())
    playlist_names = col1.multiselect(label="Select playlists you want to analyze", 
                                                options = playlists, 
                                                default = playlists[:3])

    audio_features = ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode','speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    
    audio_features_graph = col1.multiselect(label="Select the audio features you want to use in the playlist analysis", 
                                                options=audio_features, 
                                                default=audio_features)
    
    col2.plotly_chart(line_polar_playlist(df_copy, playlist_names, audio_features_graph), use_container_width= True)

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

if __name__ == "__eda_ui__":
    eda_ui()


# Agregar Streamlit UI
def eda_playlist_query(df):

    df_copy = df.copy()

    st.plotly_chart(popularidad_artista(df_copy), use_container_width= True)

    st.plotly_chart(top_artists_plot(df_copy), use_container_width= True)
        
    col1, col2 = st.columns([2, 8])
    artists = df_copy.explode('track_artists', ignore_index=True)['track_artists'].sort_values().unique()
    artists_names = col1.multiselect(label="Select playlists you want to analyze", 
                                                options = artists, 
                                                default = artists[:3])

    audio_features = ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode','speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    
    audio_features_graph = col1.multiselect(label="Select the audio features you want to use in the playlist analysis", 
                                                options=audio_features, 
                                                default=audio_features)
    
    col2.plotly_chart(line_polar_artists(df_copy, artists_names, audio_features_graph), use_container_width= True)

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

if __name__ == "__eda_playlist_query__":
    eda_playlist_query()