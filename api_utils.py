###################
##
##  YouTube API utility functions
##
##  SIADS 699 Capstone
##
##  C. Cahalan, A. Levin-Koopman, J. Olson
##
####################


import googleapiclient.discovery
import pandas as pd
import numpy as np
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

    
    # Get initial batch of results

    results = yt_client.playlistItems().list(
        playlistId = playlist_id,
        part = 'snippet',
        maxResults = max_vids
    ).execute()

    video_list = [ video['snippet']['resourceId']['videoId'] for video in results['items'] ]

    max_vids = max_vids - 50

    while ('nextPageToken' in results) and (max_vids > 0):

        # Continue pulling playlist results as long as there is a 'next page'

        results = yt_client.playlistItems().list(
            part = 'snippet',
            playlistId = playlist_id,
            pageToken = results['nextPageToken'],
            maxResults = max_vids
        ).execute()

        for video in results['items']:

            video_list.append(video['snippet']['resourceId']['videoId'])

        max_vids = max_vids - 50
        
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


def get_channel_from_vid(yt_client: object, vid_id: str) -> str:

    # Finds the channel associated with the input video

    # Parameters:
    # yt_client -- YouTube API client for requests
    # vid_id -- ID of a YouTube video

    # Returns:
    # ID string of channel associated with video

    results = yt_client.videos().list(
        part = 'snippet',
        id = vid_id
    ).execute()

    channel_id = results['items'][0]['snippet']['channelId']

    return channel_id
    


def extract_by_query(yt_client: object, query: str, max_channels: int = 50, max_vids: int = 100, excluded_channels: List = []) -> pd.DataFrame:

    # Performs data extraction from YouTube API using a keyword(s) query
    # Quota estimates use max_channels = 50, max_vids = 100

    # Parameters:
    # yt_client -- YouTube API client for requests
    # query -- A string of key words, presumably related to culinary topics
    # max_channels -- the number of channels to survey              
    # max_vids -- the number of videos to pull from each channel  
    # excluded_channels -- optional list of channel ids to exclude from results (excluded channels are still deducted from the max_channels total)  

    # Returns:
    # Pandas dataframe with channel and video features

    chan_cols = [ 'chan_query', 'chan_id', 'chan_name', 'chan_viewcount', 'chan_subcount', 'chan_start_dt', 'chan_thumb', 'chan_vidcount']
    vid_cols = ['vid_id', 'vid_name', 'vid_publish_dt', 'vid_thumb', 'vid_duration', 'vid_caption', 'vid_viewcount', 'vid_likecount', 'vid_commentcount']

    df = pd.DataFrame(columns = chan_cols + vid_cols)

    channel_results = yt_client.search().list(
        part = 'snippet',
        type = 'channel',
        q = query + ' cooking videos',
        maxResults = max_channels
    ).execute()

    # 100 quota for the search

    for channel in channel_results['items']:
        channel_id = channel['id']['channelId']

        if channel_id in excluded_channels:
            continue

        chan_info = yt_client.channels().list(
            part = ['snippet', 'contentDetails', 'statistics', 'topicDetails'],
            id = channel_id
        ).execute()['items'][0]

        # 1 quota x 50 channels = 50 quota

        chan_snip = chan_info['snippet']
        chan_det = chan_info['contentDetails']
        chan_stats = chan_info['statistics']

        # Building dataframe rows, starting with channel features.

        chan_values = [ query, channel_id, chan_snip['title'], int(chan_stats['viewCount']), int(chan_stats['subscriberCount']), chan_snip['publishedAt'], chan_snip['thumbnails']['default']['url'], int(chan_stats['videoCount']) ]

        chan_uploads_id = chan_det['relatedPlaylists']['uploads']

        # Get the id values for the channel's vids
        # 2 quota (100 vids = 2 x 50) x 50 channels = 100 quota

        # Need to catch upload errors, caused (?) by channels with no videos

        try:

            vid_ids = get_playlist(yt_client, chan_uploads_id, max_vids)

        except Exception:

            print(f"Error retrieving uploads for channel {chan_snip['title']}, ID {channel_id}.")

            continue

        for vid_id in vid_ids:

            vid_info = yt_client.videos().list(
                part = ['contentDetails', 'snippet', 'statistics'],
                id = vid_id
            ).execute()['items'][0]

            # 1 quota x 50 channels x 100 videos = 5000 quota

            vid_snip = vid_info['snippet']
            vid_det = vid_info['contentDetails']
            vid_stats = vid_info['statistics']

            # If comments are turned off, the key is missing 

            if 'commentCount' in vid_stats:
                vid_comment_count = int(vid_stats['commentCount'])
            else:
                vid_comment_count = 0

            # Key for likes can be missing 

            if 'likeCount' in vid_stats:
                vid_like_count = int(vid_stats['likeCount'])
            else:
                vid_like_count = 0

            # Key for views can be missing
                
            if 'viewCount' in vid_stats:
                vid_view_count = int(vid_stats['viewCount'])
            else:
                vid_view_count = 0

            # Finish building rows, add to dataframe

            vid_values = [ vid_id, vid_snip['title'], vid_snip['publishedAt'], vid_snip['thumbnails']['default']['url'], 
                            vid_det['duration'], vid_det['caption'], vid_view_count, vid_like_count, vid_comment_count]

            current_row = len(df.index)+1

            df.loc[current_row,:] = chan_values + vid_values


    # Total quota estimate: 100 + 50 + 100 + 5000 = 5250

    return df



