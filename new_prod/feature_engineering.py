"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import logging
import mlflow
import os.path as op

from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    load_dataset,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)

from ta_lib.data_processing.api import Outlier
from ta_lib.regression.custom_transformer import CombinedAttributesAdder

logger = logging.getLogger(__name__)


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"
    mlflow.log_param("Input Features dataset:", input_features_ds)
    mlflow.log_param("Input Target dataset:", input_target_ds)

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    cat_columns = train_X.select_dtypes("object").columns
    num_columns = train_X.select_dtypes("number").columns

    # Treating Outliers
    outlier_transformer = Outlier(method=params["outliers"]["method"])
    train_X = outlier_transformer.fit_transform(
        train_X, drop=params["outliers"]["drop"]
    )

    mlflow.log_param("outliers_dropped", params["outliers"]["drop"])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    cat_columns = train_X.select_dtypes('object').columns
    num_columns = train_X.select_dtypes('number').columns

    # housing_num_tr = num_pipeline.fit_transform(train_X, train_y)
    num_attribs = list(num_columns)

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_columns),
    ])

    housing_prepared = full_pipeline.fit_transform(train_X, train_y)


    train_X = get_dataframe(
        full_pipeline.fit_transform(train_X, train_y),
        get_feature_names_from_column_transformer(full_pipeline),
    )

    # Note: we can create a transformer/feature selector that simply drops
    # a specified set of columns. But, we don't do that here to illustrate
    # what to do when transformations don't cleanly fall into the sklearn
    # pattern.
    curated_columns = list(
        set(train_X.columns.to_list())
    )

    # saving the list of relevant columns and the pipeline.
    save_pipeline(
        curated_columns, op.abspath(op.join(artifacts_folder, "curated_columns.joblib"))
    )
    save_pipeline(
        full_pipeline, op.abspath(op.join(artifacts_folder, "features.joblib"))
    )
    mlflow.log_artifact(op.abspath(op.join(artifacts_folder, "curated_columns.joblib")))
    mlflow.log_artifact(op.abspath(op.join(artifacts_folder, "features.joblib")))
