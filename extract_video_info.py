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
import pandas as pd


def get_vid_details(url: str, path_to_subtitle_folder:str, verbose:int=0) -> Dict:
    """extracts data about a youtube video including desrition and subtitles

    Args:
        url (str): youtube video url
        path_to_subtitle_folder (str): folder to save subtitles
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
        'outtmpl': os.path.join(path_to_subtitle_folder, '%(title)s-%(id)s.%(ext)s')
    }

    with YoutubeDL(ydl_params) as ydl:
        info = ydl.extract_info(url)

    return info


def extract_subs(title:str, path_to_subs:str) -> str:
    """Extracts the saved subtitles

    Args:
        title (str): title if saved subtitle file
        path_to_subs (str): folder where subtitles are saved

    Returns:
        str: subtitles
    """
    file = [path for path in os.listdir(path_to_subs) if path.startswith(title)][0]
    subs = set()
    with open(os.path.join(path_to_subs, file), 'r') as f:
        for line in f.readlines():
            if re.search(r'^[\sa-zA-Z]+', line) is not None:
                if (not line.startswith('Kind: ')) and (not line.startswith('Language: ')) and (not line.startswith('WEB')):
                    subs.add(line)
    return [line for line in list(subs) if line.strip(' ') != '\n'][0]


def extract_by_id(video_id: str, subs_folder: str) -> pd.Series:
    """extracts info to be added to a dataframe from result from api_utils.py
    using pandas.DataFrame.apply method

    Args:
        video_id (str): id if youtube video
        subs_folder (str): folder where subtitles are saved

    Returns:
        pd.Series: resuting info from video
    """
    url = 'https://www.youtube.com/watch?v=' + video_id
    keys = [
        'description',
        'duration',
        'like_count',
        'average_rating',
        'categories',
        'tags',
        'resolution',
        'height',
        'width',
        'subtitles'
    ]
    info_dict = {key: None for key in keys}
    vid_info = get_vid_details(url, subs_folder)

    for key in info_dict.keys():
        info_dict[key] = vid_info.get(key, None)

    info_dict['subtitles'] = extract_subs(vid_info.get('title', None), subs_folder)

    return pd.Series(info_dict)


def extract_all(urls: List[str], subs_folder:str) -> pd.DataFrame:
    """extracts youtube video information

    Args:
        urls (List[str]): A list of video urls
        subs_folder (str): path to subtitles folder

    Returns:
        pd.DataFrame: info of videos
    """
    columns = [
        'id',
        'title',
        'description',
        'duration',
        'view_count',
        'like_count',
        'average_rating',
        'categories',
        'tags',
        'resolution',
        'height',
        'width',
        'subtitles'
    ]
    data_frame = pd.DataFrame(columns=columns)
    for url in urls:

        info_dict = {col: None for col in columns}

        vid_info = get_vid_details(url, subs_folder)

        for key in info_dict.keys():
            info_dict[key] = vid_info.get(key, None)

        info_dict['subtitles'] = extract_subs(info_dict['title'], subs_folder)

        
        data_frame.loc[-1] = pd.Series(info_dict)
        data_frame.reset_index(drop=True, inplace=True)

    return data_frame
