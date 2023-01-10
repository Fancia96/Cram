import json
import spotipy
import webbrowser
from spotipy import SpotifyOAuth, SpotifyException

'''

   Program uzywa rozpoznawania gestu dłoni i przełącza podłączony program spotify, wykrywa urządzenie spotify
   W związku z tym utworzony został projekt deweloperski spotify i tam dodane konta,
   które musiały zaakceptować autoryzacje żeby móc nimi sterować.
   Następnym krokiem było zaimplementowanie rozpoznawania gestu dłoni i podłączenie pod gesty akcji spotify.
   
   Obsługa spotify w pliku: spotify.py
   Obsługa gestu dłoni i wykorzystanie w tym programu spotify: music_hand_control.py

    Jak przygotwac srodowisko do uruchomienia programu:
    zajrzeć do pliku: REQUIREMENTS.TXT

    Autorzy:
     - Jakub Gwiazda(s20497)
     - Wanda Bojanowska(s18425)

    Dodatkowe załączniki: WIDEO: 2023-01-02 17-11-41.mkv

'''

#tych poniżej informacji nie udostępnie, moja własna aplikacja deweloperska spotify
# i jej dane do połączenia się z moim spotify
username = nazwauzytkownika
clientID = 'nie moge udostepnic tych danych'
clientSecret = 'nie moge udostepnic tych danych'
redirect_uri = 'http://localhost/'

#uprawnienia do konta
scope = "user-read-recently-played app-remote-control user-modify-playback-state user-read-playback-state user-read-currently-playing"
oauth_object = spotipy.SpotifyOAuth(clientID, clientSecret, redirect_uri, scope=scope)
token_dict = oauth_object.get_access_token()
token = token_dict['access_token']
spotifyObject = spotipy.Spotify(auth=token)
user_name = spotifyObject.current_user()

#pobranie id urządzenia z komputera
device_id = 0
for idx, item in enumerate(spotifyObject.devices()['devices']):
    device_id = item['id']

def next_song():
    spotifyObject.next_track(device_id)

def previous_song():
    spotifyObject.previous_track(device_id)

def start_song():
    try:
        spotifyObject.start_playback(device_id)
    except SpotifyException:
        print("Oops!  Cant start a song that is already started")

def pause_song():
    try:
        spotifyObject.pause_playback(device_id)
    except SpotifyException:
        print("Oops!  Cant pause a song that is already stopped")
