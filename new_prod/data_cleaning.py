"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""

import os
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from ta_lib.core.api import DEFAULT_ARTIFACTS_PATH

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning,
)


@register_processor("data-cleaning", "housing")
def clean_housing_table(context, params):
    """Clean the ``housing`` data table.

    The table contains information on the inventory being sold. This
    includes information on inventory id, properties of the item and
    so on.
    """

    input_dataset = "raw/housing"
    output_dataset = "cleaned/housing"
    mlflow.log_param("Products inp data path:", input_dataset)
    mlflow.log_param("Products opt dataset path:", output_dataset)

    # load dataset
    housing_df = load_dataset(context, input_dataset)
    float_cols = list(
        set(housing_df.columns.to_list()) 
        - set(['ocean_proximity'])
    )
    housing_df_clean = (
        housing_df
        # while iterating on testing, it's good to copy the dataset(or a subset)
        # as the following steps will mutate the input dataframe. The copy should be
        # removed in the production code to avoid introducing perf. bottlenecks.
        .copy()

        # set dtypes : nothing to do here
        .passthrough()

        .transform_columns(["ocean_proximity"], string_cleaning, elementwise=False)
        .replace({'': np.NaN})

        .change_type(float_cols, np.float64)
        # clean column names (comment out this line while cleaning data above)
        .clean_names(case_type='snake')
    )

    # save the dataset
    save_dataset(context, housing_df_clean, output_dataset)
    print("cleaning housing table completed")
    return housing_df_clean


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``housing`` table into ``train`` and ``test`` datasets."""

    input_dataset = "cleaned/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"
    mlflow.log_param("Input dataset path for cleaned housing:", input_dataset)
    mlflow.log_param("Train features output path:", output_train_features)
    mlflow.log_param("Train target output path:", output_train_target)
    mlflow.log_param("Test features output path:", output_test_features)
    mlflow.log_param("Test target output path:", output_test_target)

    # load dataset
    housing_df_processed = load_dataset(context, input_dataset)

    def binned_selling_price(df):
        """Bin the selling price column using quantiles."""
        return pd.qcut(df["median_house_value"], q=10)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=context.random_seed)
    housing_df_train, housing_df_test = custom_train_test_split(
        housing_df_processed, splitter, 
        by=binned_selling_price)

    target_col = "median_house_value"

    train_X, train_y = (
        housing_df_train

        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the train dataset
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # split test dataset into features and target
    test_X, test_y = (
        housing_df_test
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )
    artifacts_folder = DEFAULT_ARTIFACTS_PATH
    # save the datasets
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)
    mlflow.log_artifact(os.path.abspath(os.path.join(artifacts_folder, "../data/cleaned")))
    mlflow.log_artifact(os.path.abspath(os.path.join(artifacts_folder, "../data/train")))
    mlflow.log_artifact(os.path.abspath(os.path.join(artifacts_folder, "../data/test")))
    print("creating training datasets completed")
