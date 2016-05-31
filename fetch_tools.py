import pandas as pd
import os
import sys


def get_path_list(root_path, pattern):
    """
        Get a list of all file paths that can be seen from rooth_path
        that matches the pattern.

        :params:
            root_path - the root folder to be searched
            pattern -  regex pattern to be matched with filenames

        :return:
            path_list - list of paths that matches pattern
    """
    path_list = []
    for root, subfolders, files in os.walk(root_path):
        for filename in files:
            if pattern.match(filename):
                path_list.append(os.path.join(root, filename))

    return path_list


def get_aggregate_data(aggregate_fxn, path_list, verbose=False):
    """
        Applies aggregate_fxn to all files defined in path_list.

        :params:
            aggregate_fxn - function that aggregates the data in the
                            files defined by path_list.  Must take
                            argument (filename) from path_list and return
                            a pandas data series (data_series)
            path_list - a list of paths where aggregate_fxn will be applied

        :return:
            df_aggregate - a pandas dataframe with the collected (data_series)
            bad_paths - list of file paths where aggregate_fxn failed
    """
    df_aggregate = pd.DataFrame()
    bad_paths = []
    ii_max = len(path_list)
    for ii, file_path in enumerate(path_list):
        if verbose:
            progress_bar(ii, ii_max, ' - Processing {}'.format(file_path))
            sys.stdout.flush()
        try:
            data_series = aggregate_fxn(file_path)
            df_aggregate = df_aggregate.append(data_series, ignore_index=True)
        except:
            bad_paths.append(file_path)

    return df_aggregate, bad_paths


def progress_bar(ii, ii_max, print_str):
    """
        Simple progress bar for the terminal.

        :params:
            ii - current iteration
            ii_max - maximum iteration
            print_str - string to be printed after progress bar.
    """
    sys.stdout.write('\r')
    prog_bar = (ii*20) // ii_max
    prog_percent = ((ii+1)*100) // ii_max
    sys.stdout.write("[%-20s] %d%%. " % ('='*prog_bar, prog_percent))
    sys.stdout.write(print_str)
    sys.stdout.flush()
    return
