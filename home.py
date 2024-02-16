import streamlit as st

#Librerías para interactuar con Spotify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import MemoryCacheHandler
from bokeh.models.widgets import Div


def app_display_welcome(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, SCOPE, USERNAME):

    st.title(body="Streamlit Project Spytify 🐍")

    if "signed_in" in st.session_state and st.session_state['signed_in'] == True:
        return

    oauth = SpotifyOAuth(scope         = SCOPE, 
                         username      = USERNAME, 
                         client_id     = CLIENT_ID,
                         client_secret = CLIENT_SECRET,
                         redirect_uri  = REDIRECT_URI,
                         cache_handler = MemoryCacheHandler())
    
    # store oauth in session
    if "oauth" not in st.session_state:
        st.session_state["oauth"] = oauth

    # define temporary note
    note_temp = """
    _Note: Unfortunately, the current version of Streamlit will not allow for
    staying on the same page, so the authorization and redirection will open in a 
    new tab. This has already been addressed in a development release, so it should
    be implemented in Streamlit Cloud soon!_
    """
     # checkear si el token viene en la URL
    if "code" in st.query_params:
        st.session_state['code'] = st.query_params.code

    elif "signed_in" not in st.session_state:
        # retrieve auth url
        auth_url = oauth.get_authorize_url()

        # define welcome
        welcome_msg = """
        Welcome! :wave: This app uses the Spotify API to interact with general 
        music info and your playlists! In order to view and modify information 
        associated with any Spotify element, you must authenticate to verify you are not a robot :robot_face:. You only need to do this 
        once.
        """

        st.markdown(welcome_msg)
        st.write(" ".join(["No tokens found for this session. Please log in by",
                           "clicking the button below."]))
        
        if st.button("Click me to authenticate!"):
            # js hack to invalid the app after clicking on auth
            js_callback = "document.getElementsByClassName('stApp')[0].innerHTML = '<h1>This tab is not valid anymore. Please continue your process in the opened window.</h1>'"
            js = f"{js_callback};window.open('{auth_url}')"  # New tab or window
            html = '<img src onerror="{}">'.format(js)
            div = Div(text=html)
            st.bokeh_chart(div)


def app_get_token():
    try:
        token = get_token(st.session_state["oauth"], st.session_state["code"])
    except Exception as e:
        #st.error("An error occurred during token retrieval!")
        #st.write("The error is as follows:")
        #st.write(e)
        pass
    else:
        st.session_state["cached_token"] = token

def get_token(oauth, code):

    token = oauth.get_access_token(code, as_dict=False, check_cache=False)
    # remove cached token saved in directory
    # try:
    #     os.remove(".cache")
    # except:
    #     pass
    # return the token
    return token

def app_sign_in():
    try:
        spotify = sign_in(st.session_state["cached_token"])
    except Exception as e:
        st.error("An error occurred during sign-in!")
        st.write("The error is as follows:")
        st.write(e)
    else:
        st.session_state["signed_in"] = True
        #app_display_welcome()
        st.success("Sign in success!")
        st.info("Navigate to the sidebar to use the App.")
        
        return spotify

def sign_in(token):
    spotify = spotipy.Spotify(auth=token)
    return spotify
