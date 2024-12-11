import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from ta_lib.hyperparameter_tuning.api import objective  # Replace with your actual module name
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# Constants
DATASET_PATH = "data/raw/housing/housing.csv"
TARGET_COLUMN = "median_house_value"


@pytest.fixture
def housing_data():
    # Load the actual dataset
    data = pd.read_csv(DATASET_PATH)
    train_X = data.drop(columns=[TARGET_COLUMN])
    train_y = data[[TARGET_COLUMN]]
    return train_X, train_y


@pytest.fixture
def mock_trial():
    # Mock Optuna trial object
    from unittest.mock import MagicMock
    trial = MagicMock()
    trial.suggest_int = MagicMock(side_effect=lambda name, low, high: 10 if "n_estimators" in name else 2)
    return trial


def test_objective(housing_data, mock_trial):
    train_X, train_y = housing_data
    train_X, _, train_y, _ = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
    categorical_column = "ocean_proximity"  # Replace with the actual column name
    train_X, _, train_y, _ = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), [categorical_column])
        ],
        remainder='passthrough'  # Keep other columns unchanged
    )
    train_X_transformed = preprocessor.fit_transform(train_X)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_X_transformed, train_y), n_trials=1)
    best_trial = study.best_trial
    score = best_trial.value
    assert isinstance(score, float), "Score should be a float value"
    assert -1.0 <= score <= 1.0, "Score should be a valid R² value"