"""
Generates post-hoc model analysis using SHAP.

Expected to be run from the command line in root.

$ utilities/shap_analysis.py
"""
import os

from auto_shap import generate_shap_values
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from helpers.helpers import transform_data_with_pipeline, make_directories_if_not_exists


def make_decision_plot(expected_value: float, shap_values: np.array, x_df: pd.DataFrame, save_directory: str) -> None:
    """
    Makes a SHAP decision plot and saved it ont the shap_plots directory.

    :param expected_value: the expected value produced from the explainer
    :param shap_values: numpy array of shap values
    :param x_df: x dataframe
    :param save_directory: directory in which to share output
    """
    shap.decision_plot(expected_value, shap_values, x_df, show=False, link='identity', ignore_warnings=True)
    plt.savefig(os.path.join(save_directory, 'shap_decision_plot.png'), bbox_inches='tight')
    plt.clf()


def make_shap_interaction_plots(shap_values: np.array, x_df: pd.DataFrame, n_rank: int, save_directory: str) -> None:
    """
    Plots a series pf SHAP interaction plots fo the top n_rank features. The interaction plot shows the most meaningful
    interaction with feature.

    :param shap_values: numpy array of shap values
    :param x_df: x dataframe
    :param n_rank: number of top ranking features for which to plot interactions
    :param save_directory: directory in which to share output
    """
    for rank in range(n_rank):
        shap.dependence_plot(f'rank({rank})', shap_values, x_df, show=False)
        plt.savefig(os.path.join(save_directory, f'shap_interaction_{rank}.png'), bbox_inches='tight')
        plt.clf()


def main(model_pipeline_path, x_df_path, save_directory) -> None:
    """
    Main execution script tp produce SHAP decision and interaction plots.

    :param model_pipeline_path: path to trained modeling pipeline
    :param x_df_path: path to x predictor data
    :param save_directory: directory in which to save output, which will be created if needed
    """
    make_directories_if_not_exists([save_directory])
    pipeline = joblib.load(model_pipeline_path)
    x_df = joblib.load(x_df_path)
    model = pipeline.named_steps['model']
    x_df = transform_data_with_pipeline(pipeline, x_df)
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
    shap_values_array = shap_values_df.values
    make_decision_plot(expected_value=shap_expected_value, shap_values=shap_values_array, x_df=x_df,
                       save_directory=save_directory)
    make_shap_interaction_plots(shap_values=shap_values_array, x_df=x_df, n_rank=1, save_directory=save_directory)


if __name__ == "__main__":
    main(
        model_pipeline_path=os.path.join('modeling', 'model_results', 'xgboost', 'model', 'model.pkl'),
        x_df_path=os.path.join('modeling', 'model_results', 'xgboost', 'data', 'x_test.pkl'),
        save_directory=os.path.join('utilities', 'shap_output')
    )
