"""
This script generates the raw data that can then be transformed into data suitable for modeling. This process is
necessary before running data/generate_raw_data.py, which will create the full modeling dataframe. This script is
run independently as it takes a few minutes to complete. To pull the data, we use the popular pybaseball, which is a
nice wrapper around several baseball data sources and can be installed via pip. Purposefully, we make sequential
requests for data, though we could easily adjust the script to leverage multi-threading. This is to be considerate of
the remote servers that pybaseball hits based on the type of request.

To note, this script is expected to be run from the command line from the directory root:

$ python3 data/generate_raw_data.py

https://pypi.org/project/pybaseball/
https://pypistats.org/packages/pybaseball
"""
import datetime
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import pybaseball as pyb

from data.config import RAW_DATA_PATH, DATA_START_YEAR, DATA_END_YEAR, REMOVE_YEARS
from helpers.helpers import make_directories_if_not_exists

warnings.filterwarnings('ignore')


def _map_teams_after_2019(team_years_main_dict: dict, remove_years: list) -> dict:
    """
    pybaseball only returns the list of active teams before 2020. We assume that the teams from 2019 are valid to
    impute for future years (not expansions or moves are imminent at the time of writing). This function will
    map the current year if we are in April or after, assuming that the season will have started by this point (we
    are being optimistic and not planning for any more strikes!).

    :param team_years_main_dict: dictionary where years for which we want data are represented as keys, value are the
    teams that were present during a given year
    :param remove_years: list of years for which we do not want to find teams (e.g. strike years or 2020)
    :return: team_years_main_dict updated with years after 2020, excluding years specified in remove_years
    """
    now_timestamp = datetime.datetime.now()
    current_year = now_timestamp.year
    current_month = now_timestamp.month
    if current_month > 3:
        end_year = current_year
    else:
        end_year = current_year - 1
    years_to_add = list(np.arange(2020, end_year + 1, 1))

    teams_2019 = team_years_main_dict.get(2019)
    for year in years_to_add:
        if year not in remove_years:
            year_dict = {year: teams_2019}
            team_years_main_dict.update(year_dict)

    return team_years_main_dict


def create_years_teams_mapping(start_year: int, end_year: int, remove_years: list or None = None) -> dict:
    """
    For every year of interest, creates a list of team names that were present. The results are stored in a dictionary.
    These results are needed daily schedule and record data from pybaseball.

    :param start_year: year to start pulling the list of active teams
    :param end_year: year to stop pulling the list of active teams
    :param remove_years: list of years for which we do not want to find teams (e.g. strike years or 2020)
    :return:
    """
    team_ids_df = pyb.team_ids()
    team_ids_df = team_ids_df.loc[(team_ids_df['yearID'] >= start_year) & (team_ids_df['yearID'] <= end_year)]
    if remove_years:
        team_ids_df = team_ids_df.loc[~team_ids_df["yearID"].isin(remove_years)]
    
    unique_years = list(team_ids_df["yearID"].unique())
    team_years_main_dict = {}
    for year in unique_years:
        year_df = team_ids_df.loc[team_ids_df["yearID"] == year]
        year_teams = list(year_df["teamIDBR"].unique())
        year_teams_dict = {year: year_teams}
        team_years_main_dict.update(year_teams_dict)

    if end_year > 2019:
        team_years_main_dict = _map_teams_after_2019(team_years_main_dict, remove_years)
    return team_years_main_dict


def pull_daily_result_data(team_years_dict: dict, data_path) -> None:
    """
    Pulls daily win / loss information for every team in every year of interest. Saves the results as a compressed
    pickle file.

    :param team_years_dict: years for which we want data are represented as keys; teams that were present during a given
    year are represented as a list of values
    :param data_path: path in which to save the data
    """
    main_df = pd.DataFrame()
    for year, teams in team_years_dict.items():
        print(year)
        for team in teams:
            print(team)
            year_team_df = pyb.schedule_and_record(year, team)
            year_team_df['year'] = year
            main_df = main_df.append(year_team_df)
    joblib.dump(main_df, data_path, compress=3)


def main(data_path, start_year: int, end_year: int, remove_years: list or None = None) -> None:
    """
    Main execution script to generate raw data. The resulting file will be saved in the data_directory as 'raw.pkl'.

    :param data_path: path in which to save the output
    :param start_year: year to start pulling the list of active teams
    :param end_year: year to stop pulling the list of active teams
    :param remove_years: list of years for which we do not want to find teams (e.g. strike years or 2020)
    """
    directory_path = os.path.dirname(os.path.realpath(data_path))
    make_directories_if_not_exists([directory_path])
    team_years_dict = create_years_teams_mapping(start_year=start_year, end_year=end_year, remove_years=remove_years)
    pull_daily_result_data(team_years_dict, data_path)


if __name__ == "__main__":
    main(
        data_path=RAW_DATA_PATH,
        start_year=DATA_START_YEAR,
        end_year=DATA_END_YEAR,
        remove_years=REMOVE_YEARS
    )
