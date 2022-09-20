"""
Produces SHAP waterfall plots. This script is expected to be run from the command line.

$ python3 utilities/produce_shap_waterfall_plots.py

"""
from collections import namedtuple
import os
from typing import List

from auto_shap import generate_shap_values
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from helpers.helpers import transform_data_with_pipeline, make_directories_if_not_exists


prediction_named_tuple = namedtuple('plot_config', {'team', 'year', 'game'})
PREDICTIONS_TO_MAKE = [
    prediction_named_tuple(team='KCR', year=2022, game=20),
    prediction_named_tuple(team='KCR', year=2022, game=40),
    prediction_named_tuple(team='KCR', year=2022, game=60),
    prediction_named_tuple(team='KCR', year=2022, game=80),
    prediction_named_tuple(team='LAD', year=2022, game=20),
    prediction_named_tuple(team='LAD', year=2022, game=40),
    prediction_named_tuple(team='LAD', year=2022, game=60),
    prediction_named_tuple(team='LAD', year=2022, game=80),
    prediction_named_tuple(team='NYY', year=2022, game=20),
    prediction_named_tuple(team='NYY', year=2022, game=40),
    prediction_named_tuple(team='NYY', year=2022, game=60),
    prediction_named_tuple(team='NYY', year=2022, game=80),
    prediction_named_tuple(team='OAK', year=2022, game=20),
    prediction_named_tuple(team='OAK', year=2022, game=40),
    prediction_named_tuple(team='OAK', year=2022, game=60),
    prediction_named_tuple(team='OAK', year=2022, game=80),
]


def get_shap_values(x_df: pd.DataFrame, pipeline: Pipeline) -> tuple:
    """
    Produces SHAP values with auto_shap.

    :param x_df: x predictor dataframe
    :param pipeline: trained scikit-learn pipeline
    :return: dataframe of shap values, the shap expected value as a float, and a dataframe of global shap values
    """
    x_df = x_df.reset_index(drop=True)
    model = pipeline.named_steps['model']
    x_df = transform_data_with_pipeline(pipeline, x_df)
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
    return shap_values_df, shap_expected_value, global_shap_df


def make_waterfall_plot(shap_values_df: pd.DataFrame, shap_expected_value: float, x_df: pd.DataFrame, team: str,
                        year: int, game: int, save_directory: str) -> None:
    """
    Makes a SHAP waterfall plot for a desired team, year, and game.

    :param shap_values_df: dataframe of shap values
    :param shap_expected_value: the shap explainer's expected value
    :param x_df: x predictor dataframe
    :param team: team of interest
    :param year: year of interest
    :param game: game of interest
    :param save_directory: directory in which to save the output
    """
    x_df = x_df.loc[(x_df['team'] == team) & (x_df['year'] == year)]
    indices = x_df.index.tolist()
    shap_values_df = shap_values_df.loc[shap_values_df.index.isin(indices)]
    shap_values_df = shap_values_df.reset_index(drop=True)
    plot_index = game - 1
    shap_explanation = shap.Explanation(shap_values_df.values[plot_index], shap_expected_value,
                                        feature_names=list(shap_values_df))
    shap.waterfall_plot(shap_explanation, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f'{team}_{year}_{game}.png'))
    plt.clf()


def main(model_pipeline_path: str, modeling_df_path: str, save_directory: str,
         predictions_to_make: List[prediction_named_tuple],  drop_target: bool = True,
         years: List or None = None) -> None:
    """
    Makes SHAP waterfall plots for a desired combination of teams, years, and games.

    :param model_pipeline_path: path to load the trained pipeline
    :param modeling_df_path: path to load the modeling dataframe
    :param save_directory: directory in which to save output
    :param predictions_to_make: configuration for predictions to make
    :param drop_target: boolean of whether or not to drop the target from modeling_df
    :param years: optional list of years to subset the data to
    """
    make_directories_if_not_exists([save_directory])
    pipeline = joblib.load(model_pipeline_path)
    modeling_df = joblib.load(modeling_df_path)

    if drop_target:
        modeling_df = modeling_df.drop('target', axis=1)
    if years:
        modeling_df = modeling_df.loc[modeling_df['year'].isin(years)]

    modeling_df = modeling_df.reset_index(drop=True)
    shap_values_df, shap_expected_value, global_shap_df = get_shap_values(modeling_df, pipeline)
    for prediction in predictions_to_make:
        make_waterfall_plot(shap_values_df, shap_expected_value, modeling_df, team=prediction.team, year=prediction.year,
                            game=prediction.game, save_directory=save_directory)


if __name__ == "__main__":
    main(
        model_pipeline_path=os.path.join('modeling', 'model_results', 'xgboost', 'model', 'model.pkl'),
        modeling_df_path=os.path.join('data', 'clean', 'modeling.pkl'),
        save_directory=os.path.join('utilities', 'shap_waterfall'),
        predictions_to_make=PREDICTIONS_TO_MAKE,
        years=[2022],
        drop_target=True
    )
