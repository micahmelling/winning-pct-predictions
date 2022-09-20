"""
Config for machine learning modeling.
"""
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from collections import namedtuple
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error, 
                             median_absolute_error)


FEATURES_TO_DROP = ['year', 'team']
TARGET = 'target'
CV_SCORER = 'neg_mean_squared_error'
CV_FOLDS = 5
TEST_START_YEAR = 2019
TRAIN_END_YEAR = TEST_START_YEAR - 1
COMPLETE_HOLDOUT_YEAR = 2022
PARTITION_EVALUATION_COLS = ['winning_percentage_bin', 'pct_of_season_played_bin', 'games_and_win_pct']
PCT_OF_SEASON_PLAYED_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
WINNING_PERCENTAGE_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


FOREST_PARAM_GRID = {
    'model__max_depth': (3, 51),
    'model__min_samples_leaf': (0.001, 0.01, 'uniform'),
    'model__max_features': ['log2', 'sqrt'],
}

XGBOOST_PARAM_GRID = {
    'model__learning_rate': (0.01, 0.5, 'uniform'),
    'model__n_estimators': (75, 150),
    'model__max_depth': (3, 51),
    'model__min_child_weight': (2, 16),
}


model_named_tuple = namedtuple('model_config', {'model_name', 'model', 'param_space', 'iterations'})
MODEL_TRAINING_LIST = [
    model_named_tuple(model_name='random_forest', model=RandomForestRegressor(n_estimators=500),
                      param_space=FOREST_PARAM_GRID, iterations=20),
    model_named_tuple(model_name='extra_trees', model=ExtraTreesRegressor(n_estimators=500),
                      param_space=FOREST_PARAM_GRID, iterations=20),
    model_named_tuple(model_name='xgboost', model=XGBRegressor(n_jobs=1), param_space=XGBOOST_PARAM_GRID,
                      iterations=20),
]


evaluation_named_tuple = namedtuple('model_evaluation', {'scorer_callable', 'metric_name'})
MODEL_EVALUATION_LIST = [
    evaluation_named_tuple(scorer_callable=mean_squared_error, metric_name='mse'),
    evaluation_named_tuple(scorer_callable=median_absolute_error, metric_name='mdae'),
    evaluation_named_tuple(scorer_callable=mean_absolute_error, metric_name='mae'),
]        
