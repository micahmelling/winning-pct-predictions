"""
Function to find the best hyperparameters of a machine learning model via Bayesian Optimization.
"""
from typing import Union

import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV

from helpers.helpers import save_cv_scores, save_model


def train_model_with_bayesian_optimization(estimator: Union[Pipeline, RegressorMixin, ClassifierMixin],
                                           x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series],
                                           model_uid: str, param_space: dict, iterations: int,
                                           cv_strategy: Union[int, iter],
                                           cv_scoring: str) -> Union[Pipeline, RegressorMixin, ClassifierMixin]:
    """
    Trains a model using bayesian optimization.

    :param estimator: an estimator, such as a regression model, a classification model, or a fuller modeling pipeline
    :param x_train: x train
    :param y_train: y train
    :param model_uid: model uid
    :param param_space: parameter search space to optimize
    :param iterations: number of iterations to run the optimization
    :param cv_strategy: cross validation strategy
    :param cv_scoring: how to score cross validation folds
    :return: optimized estimator
    """
    search = BayesSearchCV(estimator, search_spaces=param_space, n_iter=iterations, scoring=cv_scoring, cv=cv_strategy,
                           n_jobs=-1, verbose=10)
    search.fit(x_train, y_train)
    best_estimator = search.best_estimator_
    save_model(best_estimator, model_uid)
    cv_results = pd.DataFrame(search.cv_results_).sort_values(by=['rank_test_score'], ascending=True)
    save_cv_scores(cv_results, model_uid)
    return best_estimator
