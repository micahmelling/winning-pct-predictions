"""
Functions to evaluate machine learning models, especially regression models.
"""
import os
from typing import Union, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from helpers.helpers import create_feature_bins, make_directories_if_not_exists
from modeling.config import evaluation_named_tuple

plt.switch_backend('Agg')


def make_predict_vs_actual_dataframe(estimator: Union[Pipeline, RegressorMixin], x_df: pd.DataFrame,
                                     target_series: Union[pd.Series]) -> pd.DataFrame:
    """
    Creates a dataframe of predictions vs. actuals

    :param estimator: estimator object
    :param x_df: predictor dataframe
    :param target_series: target values
    :return: dataframe of predictions and actuals, with the predictions stored in the 'pred' column
    """
    return pd.concat(
        [
            pd.DataFrame(estimator.predict(x_df), columns=['pred']),
            target_series.reset_index(drop=True)
        ],
        axis=1)


def make_full_predictions_dataframe(estimator: Union[Pipeline, RegressorMixin], model_uid: str, x_df: pd.DataFrame,
                                    target_series: Union[pd.Series]) -> pd.DataFrame:
    """
    Produces a dataframe consisting of a point estimate, a lower bound, an upper bound, and the actual value.

    :param estimator: estimator object
    :param model_uid: model uid
    :param x_df: predictor dataframe
    :param target_series: target values
    :returns: pandas dataframe of predictions
    """
    df = make_predict_vs_actual_dataframe(estimator, x_df, target_series)
    df = df[['pred', target_series.name]]
    x_df = x_df.reset_index(drop=True)
    df = pd.concat([df, x_df], axis=1)
    save_path = os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'predictions')
    make_directories_if_not_exists([save_path])
    df.to_csv(os.path.join(save_path, 'predictions_vs_actuals.csv'), index=False)
    return df


def _evaluate_model(target_series: pd.Series, prediction_series: pd.Series, scorer: callable, 
                    metric_name: str) -> pd.DataFrame:
    """
    Applies a scorer function to evaluate predictions against the ground-truth labels.
    
    :param target_series: target series
    :param prediction_series: prediction series
    :param scorer: scoring function to evaluate the predictions; expected to be like a scikit-learn metrics' callable
    :param metric_name: name of the metric we are using to score our model
    :returns: pandas dataframe reflecting the scoring results for the metric of interest
    """
    score = scorer(target_series, prediction_series)
    df = pd.DataFrame({metric_name: [score]})
    return df


def run_and_save_evaluation_metrics(target_series: pd.Series, prediction_series: pd.Series, model_uid: str, 
                                    evaluation_list: List[evaluation_named_tuple]) -> None:
    """
    Runs a series of evaluations metrics on a model's predictions and writes the results locally.

    :param target_series: target series
    :param prediction_series: prediction series
    :param model_uid: model uid
    :param evaluation_list: list of named tuples, which each tuple having the ordering of: the column with the
    predictions, the scoring function callable, and the name of the metric
    """
    main_df = pd.DataFrame()
    for evaluation_config in evaluation_list:
        temp_df = _evaluate_model(target_series, prediction_series,
                                  evaluation_config.scorer_callable, evaluation_config.metric_name)
        main_df = pd.concat([main_df, temp_df], axis=1)
    main_df = main_df.T
    main_df.reset_index(inplace=True)
    main_df.columns = ['scoring_metric', 'holdout_score']
    save_path = os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'evaluation_files')
    make_directories_if_not_exists([save_path])
    main_df.to_csv(os.path.join(save_path, 'evaluation_scores.csv'), index=False)


def evaluate_model_by_partition(target_series: pd.Series, prediction_series: pd.Series, x_df: pd.DataFrame,
                                cols_to_analyze: list, evaluation_list: List[evaluation_named_tuple],
                                model_uid: str) -> None:
    """
    Evaluates a machine learning model's scores on various partitions of the data. Namely, this function will analyze
    the levels of every categorical column. Therefore, to use numeric columns, it would likely be better to bin them
    into categories.

    :param target_series: target series
    :param prediction_series: predicted series
    :param x_df: x predictor dataframe
    :param cols_to_analyze: list of columns we want to analyze
    :param evaluation_list: list of named tuples, which each tuple having the ordering of: the column with the
    predictions, the scoring function callable, and the name of the metric
    :param model_uid: model uid
    """
    target_series = target_series.reset_index(drop=True)
    prediction_series = prediction_series.reset_index(drop=True)
    x_df = x_df.reset_index(drop=True)
    analysis_df = pd.concat([target_series, prediction_series, x_df], axis=1)

    main_df = pd.DataFrame()
    for col in cols_to_analyze:
        levels = list(analysis_df[col].unique())
        for level in levels:
            level_predictions_df = analysis_df.loc[analysis_df[col] == level]
            main_level_df = pd.DataFrame()
            for evaluation_config in evaluation_list:
                temp_df = _evaluate_model(
                    level_predictions_df[target_series.name],
                    level_predictions_df[prediction_series.name],
                    evaluation_config.scorer_callable,
                    evaluation_config.metric_name
                )
                main_level_df = pd.concat([main_level_df, temp_df], axis=1)

            main_level_df = main_level_df.T
            main_level_df['partition'] = f'{col}_{level}'
            main_level_df['observations'] = len(level_predictions_df)
            main_level_df.reset_index(inplace=True)
            main_level_df.rename(columns={0: 'score', 'index': 'metric'}, inplace=True)
            main_df = main_df.append(main_level_df)

    save_path = os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'evaluation_files')
    make_directories_if_not_exists([save_path])
    main_df.to_csv(os.path.join(save_path, 'evaluation_scores_by_partition.csv'), index=False)


def run_omnibus_model_evaluation(estimator: Union[Pipeline, RegressorMixin],
                                 x_df: pd.DataFrame, target_series: pd.Series, model_uid: str,
                                 evaluation_list: List[evaluation_named_tuple], target: str,
                                 partition_analyze_cols: list, pct_of_season_played_bins: list,
                                 winning_percentage_bins: list) -> None:
    """
    Runs a series of model evaluation techniques. Namely, providing scores of various metrics on the entire dataset
    and on segments of the dataset.

    :param estimator: trained regresion model or pipeline with regression model
    :param x_df: x predictor dataframe
    :param target_series: target series
    :param model_uid: model uid
    :param evaluation_list: list of named tuples, which each tuple having the ordering of: the column with the
    predictions, the scoring function callable, and the name of the metric
    :param target: name of the target
    :param partition_analyze_cols: columns for which we want to analyze the error by level
    :param pct_of_season_played_bins: bins for percent of season played
    :param winning_percentage_bins: bins for winning percentage
    """
    predictions_df = make_full_predictions_dataframe(estimator, model_uid, x_df, target_series)
    run_and_save_evaluation_metrics(predictions_df[target], predictions_df['pred'], model_uid, evaluation_list)
    x_bin_df = create_feature_bins(x_df, pct_of_season_played_bins, winning_percentage_bins)
    evaluate_model_by_partition(predictions_df[target], predictions_df['pred'], x_bin_df, partition_analyze_cols,
                                evaluation_list, model_uid)
