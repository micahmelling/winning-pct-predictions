"""
Functions to explain machine learning models.
"""
from copy import deepcopy
from functools import partial
import os
from typing import Union

from auto_shap.auto_shap import produce_shap_values_and_summary_plots
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.inspection import plot_partial_dependence
from sklearn.pipeline import Pipeline

from helpers.helpers import transform_data_with_pipeline, make_directories_if_not_exists


def produce_shap_values_and_plots(model: Union[RegressorMixin, ClassifierMixin], x_df: pd.DataFrame,
                                  model_uid: str) -> None:
    """
    Produces SHAP values and plots using the auto-shap wrapper library.

    :param model: trained regression or classification model
    :param x_df: x dataframe for which we want to produce SHAP values
    :param model_uid: model uid
    """
    save_path = os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'shap')
    make_directories_if_not_exists([save_path])
    produce_shap_values_and_summary_plots(model, x_df, save_path)


def _plot_partial_dependence(feature: str, model: Union[RegressorMixin, ClassifierMixin, Pipeline], x_df: pd.DataFrame,
                             model_uid: str) -> None:
    """
    Produces a PDP plot and saves it locally into the pdp directory.

    :param feature: name of the feature
    :param model: fitted model
    :param x_df: x dataframe
    :param model_uid: model uid
    """
    save_path = os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'pdp')
    make_directories_if_not_exists([save_path])
    _, ax = plt.subplots(ncols=1, figsize=(9, 4))
    display = plot_partial_dependence(model, x_df, [feature])
    plt.title(feature)
    plt.xlabel(feature)
    plt.savefig(os.path.join(save_path, f'{feature}.png'))
    plt.clf()


def produce_partial_dependence_plots(model: Union[RegressorMixin, ClassifierMixin, Pipeline], x_df: pd.DataFrame,
                                     model_uid: str) -> None:
    """
    Produces a PDP or ICE plot for every column in x_df. x_df is spread across all available CPUs on the machine,
    allowing plots to be created and saved in parallel.

    :param model: fitted model
    :param x_df: x dataframe
    :param model_uid: model uid
    """
    model.fitted_ = True
    pdp_plot_fn = partial(_plot_partial_dependence, model=model, x_df=x_df, model_uid=model_uid)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(pdp_plot_fn, list(x_df))


def run_omnibus_model_explanation_from_pipeline(pipeline: Pipeline, x_df: pd.DataFrame, model_uid: str) -> None:
    """
    Produces SHAP values, ICE plots, and PDP plots from a scikit-learn modeling pipeline.

    :param pipeline: trained scikit-learn pipeline
    :param x_df: x predictor dataframe
    :param model_uid: model uid
    """
    pipeline_ = deepcopy(pipeline)
    model = pipeline_.named_steps['model']
    x_df = transform_data_with_pipeline(pipeline_, x_df)
    produce_shap_values_and_plots(model, x_df, model_uid)
    produce_partial_dependence_plots(model, x_df, model_uid)
