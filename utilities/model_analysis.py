"""
Post-hoc analysis of model results for a selected model_uid. The desired model's output is looked up in
modeling/model_results. It's expected this script is run from root.

$ python3 utilities/model_analysis.py
"""

from collections import namedtuple
from typing import List
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers.helpers import make_directories_if_not_exists


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


bar_plot_named_tuple = namedtuple('plot_config', {'metric', 'partition'})
BAR_PLOTS = [
    bar_plot_named_tuple(metric='rmse', partition='winning_percentage_bin'),
    bar_plot_named_tuple(metric='mae', partition='winning_percentage_bin'),
    bar_plot_named_tuple(metric='mdae', partition='winning_percentage_bin'),
    bar_plot_named_tuple(metric='rmse', partition='pct_of_season_played_bin'),
    bar_plot_named_tuple(metric='mae', partition='pct_of_season_played_bin'),
    bar_plot_named_tuple(metric='mdae', partition='pct_of_season_played_bin'),
    bar_plot_named_tuple(metric='rmse', partition='games_and_win_pct'),
    bar_plot_named_tuple(metric='mae', partition='games_and_win_pct'),
    bar_plot_named_tuple(metric='mdae', partition='games_and_win_pct'),
]

line_plot_named_tuple = namedtuple('plot_config', {'team', 'year'})
LINE_PLOTS = [
    line_plot_named_tuple(team='KCR', year=2021),
    line_plot_named_tuple(team='LAD', year=2021),
    line_plot_named_tuple(team='HOU', year=2021),
    line_plot_named_tuple(team='LAAlop', year=2021),
    line_plot_named_tuple(team='WSN', year=2019),
    line_plot_named_tuple(team='TBR', year=2019),
    line_plot_named_tuple(team='MIA', year=2019),
    line_plot_named_tuple(team='NYY', year=2019),
]


def load_partition_errors(model_uid: str) -> pd.DataFrame:
    """
    Loads errors decomposed by different feature levels.

    :param model_uid: model uid
    :return: error partitions dataframe
    """
    return pd.read_csv(os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'evaluation_files',
                                    'evaluation_scores_by_partition.csv'))


def load_predictions(model_uid: str) -> pd.DataFrame:
    """
    Loads predictions dataframe.

    :param model_uid: model uid
    :return: error partitions dataframe
    """
    return pd.read_csv(os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'predictions',
                                    'predictions_vs_actuals.csv'))


def convert_mse_to_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts MSE to RMSE for interpretability.

    :param df: dataframe loaded by load_partition_errors
    :return: dataframe with RMSE instead of RMSE
    """
    df['score'] = np.where(
        df['metric'] == 'mse',
        np.sqrt(df['score']),
        df['score']
    )
    df['metric'] = np.where(
        df['metric'] == 'mse',
        'rmse',
        df['metric']
    )
    return df


def make_metric_bar_chart(df: pd.DataFrame, metric: str, partition: str, save_directory: str) -> None:
    """
    For a metric and partition from the data loaded by load_partition_errors(), make a barplot.

    :param df: pandas dataframe of errors by partition
    :param metric: metric of interest
    :param partition: partition of interest
    :param save_directory: directory on which to save the output
    """
    metric_df = df.loc[(df['metric'] == metric) & (df['partition'].str.startswith(partition))]
    metric_df['partition'] = metric_df['partition'].str.replace(partition, '', regex=True).str.lstrip('_')
    # for some reason, I have to call plt.tight_layout() two for the chart to look nice
    plt.tight_layout()
    plt.bar(metric_df['partition'], metric_df['score'], color='maroon')
    plt.xlabel(partition)
    plt.ylabel(metric)
    plt.xticks(rotation=90)
    plt.title(f'{metric.upper()} for {partition.title()}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f'{metric}_{partition}.png'))
    plt.clf()


def prediction_line_chart(df: pd.DataFrame, team: str, year: int, save_directory: str) -> None:
    """
    Makes a line chart of predictions for a given team and year.

    :param df: dataframe of predictions with team info metadata
    :param team: team of interest
    :param year: year of interest
    :param save_directory: directory in which to save output
    """
    prediction_df = df.loc[(df['team'] == team) & (df['year'] == year)]
    prediction_df = prediction_df.reset_index(drop=True)
    plt.plot(prediction_df['pred'], label='prediction')
    plt.plot(prediction_df['target'], label='final percentage')
    plt.plot(prediction_df['winning_percentage'], label='actual')
    # for some reason, I have to call plt.tight_layout() two for the chart to look nice
    plt.tight_layout()
    plt.ylim([0, 1])
    plt.xlabel('game')
    plt.ylabel('winning percentage')
    plt.title(f'Prediction vs. Actual for {team} {year}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f'{team}_{year}_preds.png'))
    plt.clf()


def main(model_uid: str, save_directory: str, bar_plots_config: List[bar_plot_named_tuple],
         line_plots_config: List[line_plot_named_tuple]) -> None:
    """
    Analyzes model errors and predictions using holdout predictions. Specifically, it plots error partitions and
    time-series of predictions.

    :param model_uid: model uid
    :param save_directory: directory on which to save output
    :param bar_plots_config: configuration iterable for bar plots
    :param line_plots_config: configuration for line plots
    """
    make_directories_if_not_exists([save_directory])
    partition_error_df = load_partition_errors(model_uid)
    predictions_df = load_predictions(model_uid)
    partition_error_df = partition_error_df.sort_values(by='partition')
    partition_error_df = convert_mse_to_rmse(partition_error_df)
    for bar_plot_config in bar_plots_config:
        make_metric_bar_chart(partition_error_df, metric=bar_plot_config.metric, partition=bar_plot_config.partition,
                              save_directory=save_directory)
    for line_plot_config in line_plots_config:
        prediction_line_chart(predictions_df, team=line_plot_config.team, year=line_plot_config.year,
                              save_directory=save_directory)


if __name__ == "__main__":
    main(
        model_uid='xgboost',
        save_directory=os.path.join('utilities', 'model_analysis'),
        bar_plots_config=BAR_PLOTS,
        line_plots_config=LINE_PLOTS
    )
