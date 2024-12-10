import optuna
import logging
import os.path as op
from sklearn.ensemble import RandomForestRegressor
from ta_lib.core.api import (
    load_dataset,
    load_pipeline,
    get_dataframe,
    get_feature_names_from_column_transformer,
    register_processor,
    DEFAULT_ARTIFACTS_PATH
)
from ta_lib.core.api import save_pipeline
import mlflow
logger = logging.getLogger(__name__)


def objective(trial, train_X, train_y):
    """
    Optuna objective function for RandomForestRegressor hyperparameter tuning.
    Trains a RandomForestRegressor and returns the R^2 score.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object, used to suggest hyperparameters.
    train_X : pandas.DataFrame
        The feature matrix for training the model.
    train_y : pandas.Series
        The target values corresponding to the training features.

    Returns
    -------
    float
        The R^2 score of the trained RandomForestRegressor model on the training data.
    """
    with mlflow.start_run(nested=True):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 1, 32)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        model.fit(train_X, train_y.values.ravel())
        score = model.score(train_X, train_y)
        mlflow.log_params(trial.params)
        mlflow.log_metric("score", score)
    return score


# Optimization function that runs Optuna hyperparameter optimization
def optimization(context):
    """
    Runs the hyperparameter optimization for RandomForestRegressor using Optuna.
    This function runs the objective function with Optuna to tune hyperparameters.

    Parameters
    ----------
    context : object
        Workflow context used to load datasets and pipelines.

    Returns
    -------
    RandomForestRegressor
        The trained RandomForestRegressor model with the best hyperparameters found during optimization.
    """
    artifacts_folder = DEFAULT_ARTIFACTS_PATH
    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))
    train_X_transformed = get_dataframe(
        features_transformer.fit_transform(train_X, train_y),
        get_feature_names_from_column_transformer(features_transformer),
    )
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_X_transformed, train_y), n_trials=10)
    best_trial = study.best_trial
    logger.info(f"Best hyperparameters: {best_trial.params}")
    best_model = RandomForestRegressor(
        n_estimators=best_trial.params["n_estimators"],
        max_depth=best_trial.params["max_depth"],
        min_samples_split=best_trial.params["min_samples_split"],
        min_samples_leaf=best_trial.params["min_samples_leaf"],
        random_state=42
    )
    best_model.fit(train_X_transformed, train_y.values.ravel())
    model_filename = op.join(artifacts_folder, "random_forest_model.joblib")
    save_pipeline(best_model, model_filename)

    with mlflow.start_run(nested=True):
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_score", best_trial.value)
        mlflow.log_artifact(model_filename)
    return best_model


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
