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
from sklearn import linear_model
import requests
from bs4 import BeautifulSoup
import re
import scipy.stats as stats
import urllib.request
import json
import isodate


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

def find_expans_chan(df: pd.DataFrame, min_avail: int = 50, max_on_hand: int = 200) -> List[str]:

    # Creates a list of channel ids which are appropriate candidates for the expand_channel function

    # Parameters:
    # df -- Pandas dataframe of the type returned by extract_by_query
    # min_avail -- The minimum number of available additional videos for a channel to be selected as candidate
    # max_on_hand -- Channel must have no more than this number of rows in df to be selected as candidates
    # 
    # Returns:
    # cand_list -- list of channel ids

    # With the default settings, the function returns a list of channel ids in the data base for which:
    # (1) No more than 200 of the channel's videos currently appear as rows in the dataframe and
    # (2) The channel has at least 50 more videos available on YouTube, not appearing in the dataframe yet.

    cand_list = []

    for group, frame in df.groupby('chan_id'):

        num_on_hand = len(frame)
        num_avail = frame.chan_vidcount.max() - num_on_hand

        if (num_avail >= min_avail) & (num_on_hand <= max_on_hand):

            cand_list.append(group)

    return cand_list


def expand_channel(yt_client: object, df: pd.DataFrame, channel_id: str, max_vids: int = 200) -> pd.DataFrame:

    # Extracts data for additional videos from an existing channel in a dataframe
    
    # Parameters:
    # yt_client -- a YouTube API client for making requests
    # df -- a Pandas dataframe of the type returned by extract_by_query
    # channel_id -- the id string for a YouTube channel
    # max_vids -- the maximum number of videos to extract data for

    # Returns:
    # new_df -- dataframe containing ONLY the newly extracted lines; should be concatenated with an existing dataframe, e.g. df
    
    chan_cols = [ 'chan_query', 'chan_id', 'chan_name', 'chan_viewcount', 'chan_subcount', 'chan_start_dt', 'chan_thumb', 'chan_vidcount']
    vid_cols = ['vid_id', 'vid_name', 'vid_publish_dt', 'vid_thumb', 'vid_duration', 'vid_caption', 'vid_viewcount', 'vid_likecount', 'vid_commentcount']

    new_df = pd.DataFrame(columns = chan_cols + vid_cols)

    chan_df = df[ df.chan_id == channel_id]

    chan_values = list(chan_df[chan_cols].iloc[1,:])

    old_id_list = list(chan_df.vid_id)

    full_id_list = get_uploads(yt_client, channel_id, max_vids + len(old_id_list))

    new_id_list = [ x for x in full_id_list if x not in old_id_list ]

    # Add a line for each new video, same code used in extract_by_query

    for vid_id in new_id_list:

        vid_info = yt_client.videos().list(
            part = ['contentDetails', 'snippet', 'statistics'],
            id = vid_id
        ).execute()['items'][0]

        vid_snip = vid_info['snippet']
        vid_det = vid_info['contentDetails']
        vid_stats = vid_info['statistics']

        if 'commentCount' in vid_stats:
            vid_comment_count = int(vid_stats['commentCount'])
        else:
            vid_comment_count = 0

        if 'likeCount' in vid_stats:
            vid_like_count = int(vid_stats['likeCount'])
        else:
            vid_like_count = 0
            
        if 'viewCount' in vid_stats:
            vid_view_count = int(vid_stats['viewCount'])
        else:
            vid_view_count = 0

        vid_values = [ vid_id, vid_snip['title'], vid_snip['publishedAt'], vid_snip['thumbnails']['default']['url'], 
                        vid_det['duration'], vid_det['caption'], vid_view_count, vid_like_count, vid_comment_count]

        current_row = len(new_df.index)+1

        new_df.loc[current_row,:] = chan_values + vid_values


    return new_df

def linear_pop_metric(df: pd.DataFrame, include_comments: bool = True) -> pd.DataFrame:

    # Performs a popularity metric based on a channel's "baseline" linear relationship between views and likes.

    # Parameters:
    # df -- a dataframe of the type created by extract_by_query

    # Returns:
    # out_frame -- A dataframe with an additional column 'pop_metric'

    out_frame = pd.DataFrame(columns = df.columns)

    for group, frame in df.groupby('chan_id'):

        model = linear_model.LinearRegression()

        X = np.array(frame[['vid_viewcount', 'vid_commentcount']])
        y = np.array(frame[['vid_likecount']])

        if not include_comments:

            X = X[:, [0]]

        # Fit a linear model for each channel
        
        model.fit(X,y)

        # Find the difference between actual views and predicted views, convert to z-score to normalize

        # frame.loc[:,'pop_metric'] = ( (y - model.predict(X)) / (model.predict(X)) ).flatten()

        frame.loc[:, 'log_reg_diff'] = y - model.predict(X)
        frame.loc[:, 'pop_metric'] = (frame.log_reg_diff - frame.log_reg_diff.mean()) / (frame.log_reg_diff.std())

        out_frame = pd.concat([out_frame, frame], ignore_index = True)

    return out_frame


def scrape_channel_ids(url, id_type):

    # Scrape channel IDs and channel usernames from YouTube URLs on any given website. 
    # We used ~50 websites that recommended top cooking channels

    # Parameters:
    # url: website URL as a string
    # id_type: the ID type you want to scrape; 'channel_id' or 'channel_username'

    page = requests.get(url) 
    soup = BeautifulSoup(page.content, 'html.parser')
    urls = soup.find_all('a', href=True)

    hrefs=[]
    for item in urls:
        hrefs.append(item.get('href'))

    youtube_urls=[]
    for item in hrefs:
        if id_type == 'channel_id':
            youtube_urls.append(re.findall("https://www.youtube.com/channel[^\s?]+", item))
        if id_type == 'channel_username':
            youtube_urls.append(re.findall("https://www.youtube.com/c/[^\s?]+", item))


    flat_list = [str(item.split('/')[4]) for sublist in youtube_urls for item in sublist]

    return flat_list

def get_channel_stats(youtube, channel_id, id_type):
    
    # Get the YouTube API response based on the IDs you scrape using scrape_channel_ids

    # Parameters:
    # youtube: result of make_client
    # channel_id: scraped ID
    # id_type: 'channel_id' or 'channel_username'

    if id_type == 'channel_id':
        request = youtube.channels().list(
            part = 'snippet,contentDetails,statistics',
            id=channel_id
        )

        response = request.execute()
  
    if id_type == 'channel_username':
        request = youtube.channels().list(
        part = 'snippet,contentDetails,statistics',
        forUsername=channel_id
        )

        response = request.execute()
    
    return response['items']

def get_video_list(youtube, upload_id):

    # Get a channel's videos based on the upload id

    # Parameters:
    # youtube: result of make_client
    # upload_id:  channel_stats[0]['contentDetails']['relatedPlaylists']['uploads']

    video_list = []
    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=upload_id,
        maxResults=50
    )
    next_page = True
    while next_page:
        response = request.execute()
        data = response['items']

        for video in data:
            video_id = video['contentDetails']['videoId']
            if video_id not in video_list:
                video_list.append(video_id)

        if 'nextPageToken' in response.keys():
            next_page = True
            request = youtube.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=upload_id,
                pageToken=response['nextPageToken'],
                maxResults=50
            )
        else:
            next_page = False

    return video_list

def get_video_details(youtube, video_list):

    # Extract the columns needed for the final data frame based on the video list

    # Parameters:
    # youtube: result of make_client
    # video_list: videos from get_video_list

    stats_list=[]
    for i in range(0, len(video_list), 50):
        request= youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_list[i:i+50]
        )

        data = request.execute()
        for video in data['items']:
            chan_id = video['snippet']['channelId']
            vid_id = video['id']
            vid_name = video['snippet']['title']
            vid_publish_dt = video['snippet']['publishedAt']
            vid_thumb = video['snippet']['thumbnails']['default']['url']
            vid_duration = video['contentDetails']['duration']
            vid_caption = video['contentDetails']['caption']
            vid_viewcount = video['statistics'].get('viewCount',0)
            vid_likecount = video['statistics'].get('likeCount',0)
            vid_commentcount = video['statistics'].get('commentCount',0)
            data_dict=dict(chan_id=chan_id, vid_id=vid_id, vid_name=vid_name, vid_publish_dt=vid_publish_dt,
                          vid_thumb=vid_thumb,vid_duration=vid_duration,vid_caption=vid_caption,vid_viewcount=vid_viewcount,
                          vid_likecount=vid_likecount,vid_commentcount=vid_commentcount)
            stats_list.append(data_dict)

    return stats_list

def create_video_df(youtube, channel_ids, id_type):

    # Input a list of channel IDs, and this function outputs all videos for those channel IDs in the format needed for this project

    # Parameters:
    # youtube: result of make_client
    # channel_ids: list of channel IDs
    # id_type: 'channel_id' or 'channel_username' 

    channel_dfs = []
    vid_dfs = []
    for channel_id in channel_ids:
        try:
            channel_stats = get_channel_stats(youtube, channel_id, id_type)

            channel_dfs.append(pd.json_normalize(channel_stats))

            upload_id = channel_stats[0]['contentDetails']['relatedPlaylists']['uploads']
            video_list = get_video_list(youtube, upload_id)

            video_data = get_video_details(youtube, video_list)
            vid_dfs.append(pd.json_normalize(video_data))
        except:
            pass

    channel_df = pd.concat(channel_dfs)

    vid_df = pd.concat(vid_dfs)

    channel_df = channel_df.rename(columns={'id':'chan_id','snippet.title':'chan_name','statistics.viewCount':'chan_viewcount',
                                            'statistics.subscriberCount':'chan_subcount','snippet.publishedAt':'chan_start_dt',
                                            'snippet.thumbnails.default.url':'chan_thumb','statistics.videoCount':'chan_vidcount'})
  
    channel_df = channel_df[['chan_id','chan_name','chan_viewcount','chan_subcount','chan_start_dt','chan_thumb','chan_vidcount']]

    final_df = vid_df.merge(channel_df, how='left', on='chan_id')

    column_order = ['chan_id','chan_name','chan_viewcount','chan_subcount','chan_start_dt','chan_thumb','chan_vidcount',
                    'vid_id','vid_name','vid_publish_dt','vid_thumb','vid_duration','vid_caption','vid_viewcount','vid_likecount','vid_commentcount']
    
    return final_df[column_order]

def remove_unqualified_videos(df):

    # data cleaning for videos
    # remove videos<60 seconds and any that contain '#shorts' in the title
    # also drop duplicates based on the vid_id

    #parameters: dataframe that you are looking to clean


    df = df[df['vid_duration'].notna()]
    df = df.drop_duplicates(subset='vid_id', keep="first")
    df = df[~df['vid_name'].str.contains('#shorts')]
    df['vid_seconds'] = df['vid_duration'].apply(lambda x: isodate.parse_duration(x).total_seconds())
    df = df[df['vid_seconds']>60]
    return df.drop(columns=['vid_seconds']).reset_index(drop=True)
