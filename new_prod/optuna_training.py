import logging
import os.path as op
from ta_lib.hyperparameter_tuning.api import optimization
from ta_lib.core.api import (
    load_dataset,
    load_pipeline,
    get_dataframe,
    get_feature_names_from_column_transformer,
    register_processor,
    DEFAULT_ARTIFACTS_PATH
)
logger = logging.getLogger(__name__)


@register_processor("optuna", "train-model-optuna")
def train_model(context, params):
    """Train a regression model using Optuna hyperparameter tuning."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))
    train_X_transformed = get_dataframe(  # noqa: F841
        features_transformer.fit_transform(train_X, train_y.values.ravel()),
        get_feature_names_from_column_transformer(features_transformer),
    )
    best_model = optimization(context)
    
    logger.info(f"Best model parameters: {best_model.get_params()}")
    return best_model
