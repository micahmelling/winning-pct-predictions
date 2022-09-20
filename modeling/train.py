"""
Script to train and evaluate a series of models for our project.

This script is expected to be run from root on the command line.

$ python3 modeling/train.py
"""

from time import time
from typing import List, Union

import joblib
import pandas as pd

from data.config import CLEAN_DATA_PATH, DATA_START_YEAR
from helpers.helpers import prep_modeling_and_evaluation_data, save_modeling_data_in_model_directory, \
    create_custom_ts_cv_splits
from modeling.config import model_named_tuple, evaluation_named_tuple, TARGET, MODEL_EVALUATION_LIST, \
    MODEL_TRAINING_LIST, TEST_START_YEAR, CV_SCORER, TRAIN_END_YEAR, CV_FOLDS, PARTITION_EVALUATION_COLS, \
    PCT_OF_SEASON_PLAYED_BINS, WINNING_PERCENTAGE_BINS, COMPLETE_HOLDOUT_YEAR
from modeling.model import train_model_with_bayesian_optimization
from modeling.pipeline import get_pipeline
from modeling.evaluate import run_omnibus_model_evaluation
from modeling.explain import run_omnibus_model_explanation_from_pipeline


def create_training_testing_data(cleaned_data_path: str, test_year_start: int, target: str,
                                 complete_holdout_year: int or None = None) -> tuple:
    """
    Creates the training and testing data for our modeling problem.

    :param cleaned_data_path: path to cleaned data for modeling
    :param test_year_start: year to start the test set period
    :param target: name of the target column
    :param complete_holdout_year: optional year to completely eliminate from training and testing; most often, this
    will be the current year
    :return: x train, y train, x test, and y test
    """
    modeling_df = joblib.load(cleaned_data_path)
    modeling_df = modeling_df.dropna()
    modeling_df = modeling_df.reset_index(drop=True)
    modeling_df['year'] = modeling_df['year'].astype(int)
    if complete_holdout_year:
        modeling_df = modeling_df.loc[modeling_df['year'] < complete_holdout_year]
    x_train, y_train, x_test, y_test = prep_modeling_and_evaluation_data(modeling_df, test_year_start, target)
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return x_train, y_train, x_test, y_test
    

def train_and_evaluate_explain_models(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                                      y_test: pd.Series, models_list: List[model_named_tuple],
                                      cv_strategy: Union[int, iter], cv_scoring: str,
                                      model_evaluation_list: List[evaluation_named_tuple], target: str,
                                      partition_analyze_cols: list, games_played_bins: list,
                                      winning_percentage_bins: list) -> None:
    """
    Trains, evaluates, and explains a series of machine learning models.

    :param x_train: x train
    :param y_train: y train
    :param x_test: x test
    :param y_test: y test
    :param models_list: list of named tuples, with each tuple having the ordering of the model name, the model class,
    the parameter optimization space, and the number of iterations to train the model
    :param cv_strategy: cross validation strategy
    :param cv_scoring: cross validation scoring strategy
    :param model_evaluation_list: list of named tuples, which each tuple having the ordering of: the column with the
    predictions, the scoring function callable, and the name of the metric
    :param target: name of the target
    :param partition_analyze_cols: columns for which we want to analyze the error by level
    :param games_played_bins: bins for games played
    :param winning_percentage_bins: bins for winning percentage
    """
    for model in models_list:
        print(f'training {model.model_name}...')
        save_modeling_data_in_model_directory(x_train, y_train, x_test, y_test, model.model_name)
        pipeline = get_pipeline(model.model)
        best_model = train_model_with_bayesian_optimization(
            estimator=pipeline,
            x_train=x_train,
            y_train=y_train,
            model_uid=model.model_name,
            param_space=model.param_space,
            iterations=model.iterations,
            cv_strategy=cv_strategy,
            cv_scoring=cv_scoring
        )
        run_omnibus_model_evaluation(
            estimator=best_model,
            x_df=x_test,
            target_series=y_test,
            model_uid=model.model_name,
            evaluation_list=model_evaluation_list,
            target=target,
            partition_analyze_cols=partition_analyze_cols,
            pct_of_season_played_bins=games_played_bins,
            winning_percentage_bins=winning_percentage_bins
        )
        run_omnibus_model_explanation_from_pipeline(
            pipeline=best_model,
            x_df=x_test,
            model_uid=model.model_name
        )


def main(cleaned_data_path: str, test_year_start: int, target: str, models_list: List[model_named_tuple],
         cv_scoring: str, train_start_year: int, train_end_year: int, cv_folds: int,
         model_evaluation_list: List[evaluation_named_tuple], partition_analyze_cols: list, games_played_bins: list,
         winning_percentage_bins: list, complete_holdout_year: int or None = None):
    """
    Main execution function to train and evaluate models.

    :param cleaned_data_path: path to cleaned data for modeling
    :param test_year_start: year to start the test set period
    :param target: name of the target column
    :param models_list: list of named tuples, with each tuple having the ordering of the model name, the model class,
    the parameter optimization space, and the number of iterations to train the model
    :param cv_scoring: cross validation scoring strategy
    :param train_start_year: year training data starts
    :param train_end_year: year training data ends
    :param cv_folds: number of cross validation folds
    :param model_evaluation_list: list of named tuples, which each tuple having the ordering of: the column with the
    predictions, the scoring function callable, and the name of the metric
    :param partition_analyze_cols: columns for which we want to analyze the error by level
    :param games_played_bins: bins for games played
    :param winning_percentage_bins: bins for winning percentage
    :param complete_holdout_year: optional year to completely eliminate from training and testing; most often, this
    will be the current year
    """
    x_train, y_train, x_test, y_test = create_training_testing_data(
        cleaned_data_path=cleaned_data_path,
        test_year_start=test_year_start,
        target=target,
        complete_holdout_year=complete_holdout_year
    )
    cv_splits = create_custom_ts_cv_splits(
        df=x_train,
        start_year=train_start_year,
        end_year=train_end_year,
        cv_folds=cv_folds
    )
    train_and_evaluate_explain_models(
        x_train,
        y_train,
        x_test,
        y_test,
        models_list,
        cv_strategy=cv_splits,
        cv_scoring=cv_scoring,
        model_evaluation_list=model_evaluation_list,
        target=target,
        partition_analyze_cols=partition_analyze_cols,
        games_played_bins=games_played_bins,
        winning_percentage_bins=winning_percentage_bins
    )


if __name__ == "__main__":
    script_start_time = time()
    main(
        cleaned_data_path=CLEAN_DATA_PATH,
        test_year_start=TEST_START_YEAR,
        target=TARGET,
        models_list=MODEL_TRAINING_LIST,
        cv_scoring=CV_SCORER,
        train_start_year=DATA_START_YEAR,
        train_end_year=TRAIN_END_YEAR,
        cv_folds=CV_FOLDS,
        model_evaluation_list=MODEL_EVALUATION_LIST,
        partition_analyze_cols=PARTITION_EVALUATION_COLS,
        games_played_bins=PCT_OF_SEASON_PLAYED_BINS,
        winning_percentage_bins=WINNING_PERCENTAGE_BINS,
        complete_holdout_year=COMPLETE_HOLDOUT_YEAR
    )
    print(f"--- {time() - script_start_time} seconds for script to run ---")
