"""
Various helper functions.
"""
from copy import deepcopy
import os
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.pipeline import Pipeline


def make_directories_if_not_exists(directories_list: list) -> None:
    """
    Makes directories in the current working directory if they do not exist.

    :param directories_list: list of directories to create
    """
    for directory in directories_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


def drop_features(df: Union[pd.DataFrame, np.ndarray], features_list: list) -> pd.DataFrame:
    """
    Drops features (columns) from a dataframe.

    :param df: pandas dataframe
    :param features_list: list of features (columns) to drop
    :return: dataframe with dropped columns
    """
    df = df.drop(features_list, axis=1, errors='ignore')
    return df


def save_cv_scores(df: pd.DataFrame, model_uid: str) -> None:
    """
    Saves cross-validation scores into a model-specific directory.

    :param df: dataframe of cross-validation scores
    :param model_uid: model uid
    """
    save_directory = os.path.join('modeling', 'model_results', model_uid, 'cv_scores')
    make_directories_if_not_exists([save_directory])
    df.to_csv(os.path.join(save_directory, 'cv_scores.csv'), index=False)


def save_model(model: Union[RegressorMixin, ClassifierMixin], model_uid: str,
               model_append_name: str = None, ) -> None:
    """
    Saves a pickled model into a model-specific directory.

    :param model: regression or classification model
    :param model_uid: model uid
    :param model_append_name: name to append to the model name
    """
    model_name = 'model'
    if model_append_name:
        model_name = f'{model_name}_{model_append_name}'
    save_directory = os.path.join('modeling', 'model_results', model_uid, 'model')
    make_directories_if_not_exists([save_directory])
    joblib.dump(model, os.path.join(save_directory, f'{model_name}.pkl'))


def save_modeling_data_in_model_directory(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                                          y_test: pd.Series, model_uid: str) -> None:
    """
    Saves modeling data into a model-specific directory.

    :param x_train: x train
    :param y_train: y train
    :param x_test: x test
    :param y_test: y test
    :param model_uid: model uid
    """
    save_directory = os.path.join('modeling', 'model_results', model_uid, 'data')
    make_directories_if_not_exists([save_directory])
    joblib.dump(x_train, os.path.join(save_directory, 'x_train.pkl'))
    joblib.dump(y_train, os.path.join(save_directory, 'y_train.pkl'))
    joblib.dump(x_test, os.path.join(save_directory, 'x_test.pkl'))
    joblib.dump(y_test, os.path.join(save_directory, 'y_test.pkl'))


def create_data_splits_on_year(df: pd.DataFrame, test_year_start: int) -> tuple:
    """
    Creates two splits of data: training and testing for model training and evaluation based on a year-wise split.

    :param df: pandas dataframe on which we can subset based on a year column
    :param test_year_start: year to start the testing data
    :return: training dataframe and testing dataframe
    """
    train_df = df.loc[df['year'] < test_year_start]
    test_df = df.loc[df['year'] >= test_year_start]
    return train_df, test_df


def create_x_y_split(df: pd.DataFrame, target: str) -> tuple:
    """
    Creates an x-y split for modeling.

    :param df: dataframe of modeling data
    :param target: name of the target column
    :return: x predictor dataframe and y target series
    """
    y_series = df[target]
    x_df = df.drop(target, axis=1)
    return x_df, y_series


def prep_modeling_and_evaluation_data(df: pd.DataFrame, test_year_start: int, target: str) -> tuple:
    """
    Prepares data for modeling, evaluation, and prediction by creating necessary target and predictor splits.

    :param df: dataframe of modeling data
    :param test_year_start: year to start the testing data
    :param target: name of the target column
    :return: x train, y train, x test, y test, and a prediction dataframe
    """
    train_df, test_df = create_data_splits_on_year(df, test_year_start)
    x_train, y_train = create_x_y_split(train_df, target)
    x_test, y_test = create_x_y_split(test_df, target)
    return x_train, y_train, x_test, y_test


def create_custom_ts_cv_splits(df: pd.DataFrame, start_year: int, end_year: int, cv_folds: int) -> list:
    """
    Creates a set of custom cross validation splits based on year. The function takes a start year and an end year along
    with a number of cv folds. Based on the number of years between the provided years, it will create an equal number
    of array splits based on cv folds. The folds are arranged as a time-series cross validation problem. That is,
    in the first split, the first split is the training data and the second split is the testing data. In the second
    split, the first two splits are the training data, and the third split is the testing data. And so on.

    :param df: pandas dataframe of training data
    :param start_year: start year of the cross validation folds
    :param end_year: end year of the cross validation folds
    :param cv_folds: number of cv folds
    :return: list of tuples, with each tuple containing two items - the first is the index of the training observations
    and the second is the index of the testing observations
    """
    cv_splits = []
    years = list(np.arange(start_year, end_year + 1, 1))
    year_splits = np.array_split(years, cv_folds)
    for n, year_split in enumerate(year_splits):
        if n != cv_folds - 1:
            train_ids = year_splits[:n + 1]
            train_ids = np.concatenate(train_ids)
            test_ids = year_splits[n + 1]
            train_indices = df.loc[df['year'].isin(train_ids)].index.values.astype(int)
            test_indices = df.loc[df['year'].isin(test_ids)].index.values.astype(int)
            cv_splits.append((train_indices, test_indices))
    return cv_splits


def transform_data_with_pipeline(pipeline: Pipeline, x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms x_df with the pre-processing steps defined in the pipeline.

    :param pipeline: scikit-learn pipeline
    :param x_df: x predictor dataframe for transformation
    :return: transformed x dataframe
    """
    pipeline_ = deepcopy(pipeline)
    pipeline_.steps.pop(len(pipeline_) - 1)
    x_df = pipeline_.transform(x_df)
    return x_df


def create_feature_bins(df: pd.DataFrame, pct_of_season_played_bins: list, winning_percentage_bins: list):
    """
    Creates bins for a two pre-defined features: games_played and winning_percentage. The function also creates
    a third new features that concatenates games_played_bin and winning_percentage_bin

    :param df: modeling dataframe
    :param pct_of_season_played_bins: bins for percent of season played
    :param winning_percentage_bins: bins for winning percentage
    :return: dataframe with two new binned features: games_played_bin and winning_percentage_bin and a concatenation of
    the two called games+win_pct
    """
    df_copy = deepcopy(df)
    df_copy['pct_of_season_played_bin'] = pd.cut(df_copy['pct_of_season_played'], bins=pct_of_season_played_bins,
                                                 include_lowest=True)
    df_copy['pct_of_season_played_bin'] = df_copy['pct_of_season_played_bin'].astype(str)
    df_copy['winning_percentage_bin'] = pd.cut(df_copy['winning_percentage'], bins=winning_percentage_bins,
                                               include_lowest=True)
    df_copy['winning_percentage_bin'] = df_copy['winning_percentage_bin'].astype(str)
    df_copy['games_and_win_pct'] = df_copy['pct_of_season_played_bin'] + '_' + df_copy['winning_percentage_bin']
    return df_copy
