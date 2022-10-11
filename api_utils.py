import googleapiclient.discovery

from typing import List, Set, Dict, Tuple, Optional

def make_client(api_key: str) -> object:

    # Creates a YouTube API client for use in subsequent requests
    # User's API key is only needed once to create the client

    yt_client = googleapiclient.discovery.build('youtube', 'v3', developerKey = api_key)

    return yt_client


def get_playlist(yt_client: object, playlist_id: str, max_vids: int = 50) -> List[str]:

    # Parameters:
    # yt_client -- YouTube API client for requests
    # playlist_id -- ID of a YouTube playlist
    # max_vids -- Number of videos to return, if available (50 allowed by YouTube)

    # Returns:
    # List of video ID strings

    results = yt_client.playlistItems().list(
        playlistId = playlist_id,
        part = ['contentDetails'],
        maxResults = max_vids
    ).execute()
    
    video_list = [ video['contentDetails']['videoId'] for video in results['items'] ]

    return video_list        


def get_uploads(yt_client: object, channel_id: str, max_vids: int = 50) -> List[str]:

    # Finds the most recent uploads associated with a YouTube channel

    # Parameters:
    # yt_client -- YouTube API client for requests
    # channel_id -- ID of a YouTube channel
    # max_vids -- Number of videos to return, if available (50 allowed by YouTube)

    # Returns:
    # List of video ID strings

    results = yt_client.channels().list(
        part='contentDetails',
        id = channel_id,
    ).execute()

    upload_id = results['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    upload_list = get_playlist(yt_client, upload_id, max_vids = max_vids)

    return upload_list
