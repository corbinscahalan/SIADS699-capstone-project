###################
##
##  youtube-dl video extraction functions
##
##  SIADS 699 Capstone
##
##  C. Cahalan, A. Levin-Koopman, J. Olson
##
####################

from youtube_dl import YoutubeDL
import pandas as pd
from typing import List, Dict
import os
import re



def get_vid_details(url: str, path_to_thumbs:str, verbose:int=0) -> Dict:
    """extracts data about a youtube video including desrition and subtitles

    Args:
        url (str): youtube video url
        path_to_thumbs (str): folder to save thumbnails and temp subtitle directory
        verbose (int): whether to show youtube-dl output in terminal, 0 no output, 1 base, > 1 all info

    Returns:
        Dict: info dictionary about the video
    """

    ydl_params = {
        'verbose': True if verbose > 1 else False,
        'quiet': True if verbose == 0 else False,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'skip_download': True,
        'subtitleslangs': ['en',],
        'writethumbnail': True,
        'nooverwrites': True,
        'outtmpl': os.path.join(path_to_thumbs, '%(id)s.%(ext)s')
    }

    with YoutubeDL(ydl_params) as ydl:
        info = ydl.extract_info(url)

    return info


def extract_subs(sub_filename:str, path_to_subs:str) -> List[str]:
    """Extracts the saved subtitles

    Args:
        title (str): title if saved subtitle file
        path_to_subs (str): folder where subtitles are saved

    Returns:
        List[str]: subtitles
    """
    
    file = os.path.join(path_to_subs, sub_filename)
    subs = []
    with open(file, 'r') as f:
        for line in f.readlines():
            
            
            if ('-->' not in line) and ('</c>' not in line) and (line.strip() != '\n') and (not line.startswith('Kind: ')) and (not line.startswith('Language: ')) and (not line.startswith('WEB')) and (not line.startswith('0')):
                if line not in subs:
                    subs.append(line)
                else:
                    pass
    os.remove(file)
    return [line for line in subs if line.strip(' ') != '\n']


def extract_by_id(video_id:str, thumb_folder:str, ydl_verbose:int=0) -> pd.Series:
    """extracts info to be added to a dataframe from result from api_utils.py
    using pandas.DataFrame.apply method

    Args:
        video_id (str): id if youtube video
        thumb_folder (str): folder to save thumbnails and temp subtitles
        ydl_verbose (int): whether to show youtube-dl output in terminal, 0 no output, 1 base, > 1 all info

    Returns:
        pd.Series: resuting info from video
    """
    url = 'https://www.youtube.com/watch?v=' + video_id
    keys = [
        # 'id',
        # 'title',
        # 'formats',
        # 'thumbnails',
        'description',
        # 'upload_date',
        # 'uploader',
        # 'uploader_id',
        # 'uploader_url',
        # 'channel_id',
        # 'channel_url',
        'duration',
        # 'view_count',
        # 'average_rating',
        'age_limit',
        # 'webpage_url',
        'categories',
        'tags',
        'is_live',
        # 'like_count',
        # 'channel',
        # 'extractor',
        # 'webpage_url_basename',
        # 'extractor_key',
        # 'playlist',
        # 'playlist_index',
        # 'thumbnail',
        # 'display_id',
        # 'requested_subtitles',
        # 'requested_formats',
        # 'format',
        # 'format_id',
        'width',
        'height',
        # 'resolution',
        'fps',
        'vcodec',
        'vbr',
        # 'stretched_ratio',
        'acodec',
        'abr',
        # 'ext'
    ]
    info_dict = {key: None for key in keys}
    
    try:
        vid_info = get_vid_details(url, thumb_folder, ydl_verbose)
    except Exception:
        return pd.Series(info_dict)

    for key in info_dict.keys():
        info_dict[key] = vid_info.get(key, None)

    thumb_name = os.path.split([entry.get('filename') for entry in vid_info['thumbnails'] if entry.get('filename', None) is not None][0])[1]

    info_dict['thumb_name'] = thumb_name

    
    try:
        sub_ext = '.en.' + vid_info['requested_subtitles']['en']['ext']
        info_dict['subtitles'] = extract_subs(video_id + sub_ext, thumb_folder)
    except Exception:
        info_dict['subtitles'] = ''


    return pd.Series(info_dict)

