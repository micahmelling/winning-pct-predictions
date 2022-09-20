"""
Function to create a scikit-learn modeling pipeline.
"""

from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from helpers.helpers import drop_features
from modeling.config import FEATURES_TO_DROP


def get_pipeline(model: RegressorMixin or ClassifierMixin) -> Pipeline:
    """
    Creates a scikit-learn modeling pipeline for our modeling problem. In this case, a set of features can be dropped
    per the FEATURES_TO_DROP global defined in modeling.config. A model is then applied.

    :param model: regression or classification model
    :return: scikit-learn pipeline
    """
    pipeline = Pipeline(steps=[
        ('dropper', FunctionTransformer(drop_features, validate=False,
                                        kw_args={
                                            'features_list': FEATURES_TO_DROP
                                        })),
        ('model', model)
        ])
    return pipeline
