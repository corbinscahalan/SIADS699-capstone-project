import pandas as pd
import os
from api_utils import make_client, extract_by_query
from googleapiclient.errors import HttpError






def extract_query(api_key:str, query:str, excluded_chanels:set, compiled_data=None):
    """extracts a dataframe of query results

    Args:
        api_key (str): api key as a string
        query (str): the query for the search
        excluded_chanels (set): a set of channel ids to exclude from the results
        compiled_data (None or pd.DataFrame, optional): data to append results to. Defaults to None.

    Returns:
        tuple(pd.DataFrame, set): updated dataframe with new query results, updated set of excluded channels
    """
    client = make_client(api_key)
    query = query.replace('/', ' ')
    print('Searching query:', query.rstrip('\n'))
    df = extract_by_query(client, query.rstrip('\n'), excluded_channels=list(excluded_chanels), max_channels=50, max_vids=50)
    if compiled_data is None:
        compiled_data = df
    else:
        compiled_data = pd.concat((compiled_data, df), axis=0, ignore_index=True)
    exclude = set(df['chan_id'].unique())
    excluded_chanels = excluded_chanels.union(exclude)
    return compiled_data, excluded_chanels




# for use in terminal
def get_input(query):
    """uses terminal to use, skip or modify a query

    Args:
        query (str): the query to inspect

    Returns:
        str: new query
    """
    
    while True:
        print()
        user_input = input(f'Query: {query}\nto use: press enter, to skip: enter -s, to modify: enter -m new_query\n==> ')
        if user_input == '':
            return query
        elif user_input == '-s':
            return None
        elif user_input.split()[0].strip() == '-m' and user_input.split()[1:] != []:

            ret_query = ' '.join(user_input.split()[1:])
            return ret_query

        else:
            print('Not recognized try again')
            continue

def api_gen(apis:list):
    """_summary_

    Args:
        apis (list): list of api keys as strings

    Yields:
        str: api key
    """
    for api in apis:
        yield api


def extract_all(api_key_list:list, query_list:list, excluded_chanels: set, compiled_data=None, with_terminal=False, intermediate_save_folder=None):
    """extracts queries based on a query list and a list of api keys

    Args:
        api_key_list (list): list of api keys as strings
        query_list (list): a list of querries
        excluded_chanels (set): a set of channel ids to exclude from the search results
        compiled_data (None or pd.DataFrame, optional): data to append results to. Defaults to None.
        with_terminal (bool, optional): whether to inspect each query and decide to modify or skip only for use with a terminal. Defaults to False.
        intermediate_save_folder (None, or str, optional): folder to save results after each query. Defaults to None.

    Returns:
        tuple(pd.DataFrame, set): updated dataframe with new query results, updated set of excluded channels
    """



    api_keys = api_gen(api_key_list)
    api_string = next(api_keys)
    for query in query_list:

        
        if with_terminal:
            query = get_input(query.rstrip('\n').replace('/', ' '))
            if query is None:
                continue
        
        try:
            compiled_data, excluded_chanels = extract_query(api_string, query=query, excluded_chanels=excluded_chanels, compiled_data=compiled_data)
            print('Data Shape:', compiled_data.shape, 'Excluded Channel List len:', len(excluded_chanels))
        except HttpError:
            try:
                api_string = next(api_keys)
                print('='*50)
                print('Query limit reached trying next api key')
                compiled_data, excluded_chanels = extract_query(api_string, query=query, excluded_chanels=excluded_chanels, compiled_data=compiled_data)
                print('Data Shape:', compiled_data.shape, 'Excluded Channel List len:', len(excluded_chanels))
            except StopIteration:
                print(f'Final api ran out on query: {query}, not included in data')
                return compiled_data, excluded_chanels
        
        if intermediate_save_folder is not None:
            compiled_data.to_csv(os.path.join(intermediate_save_folder, f'compiled_data_through_query_{query}.csv'), index=False)
            excluded = pd.Series(list(excluded_chanels))
            excluded.to_csv(os.path.join(intermediate_save_folder, f'excluded_channels_through_query_{query}.csv'), index=False)
            


    return compiled_data, excluded_chanels


if __name__ == '__main__':

    # list of api strings
    api_list = []

    # list of query terms
    new_foods_text_file = ''
    start_index = 180
    stop_index = 200

    query_list = []
    with open(new_foods_text_file, 'r') as file:
        query_list = file.readlines()[start_index: stop_index]

    # excluded channels as a set
    excluded_channel_csv_file = ''
    exclude_list = pd.read_csv(excluded_channel_csv_file) # this should be a pandas Series
    excluded_channels = set(exclude_list.values.ravel())

    # data so far if starting from scratch set to None
    data_file = ''
    data = pd.read_csv(data_file) if data_file != '' else None

    # where to save intermediate results (after each query) set to None otherwise
    save_folder = ''

    # extract data
    finished_data, excluded_channel_set = extract_all(api_list, query_list, excluded_channels, data, with_terminal=True, intermediate_save_folder=save_folder if save_folder != '' else None)

    # save results to csv
    results_save_name = 'extracted_data.csv'
    finished_data.to_csv(os.path.join(save_folder, results_save_name), index=False)
    
    # save excluded channels to csv
    excluded_channels_save_name = 'excluded_channels.csv'
    excluded_channels = pd.Series(list(excluded_channels))
    excluded_channels.to_csv(os.path.join(save_folder, excluded_channels_save_name), index=False)