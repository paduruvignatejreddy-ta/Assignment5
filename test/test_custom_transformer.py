import pytest
import numpy as np
from ta_lib.regression.custom_transformer import CombinedAttributesAdder


@pytest.fixture
def sample_data():
    # Create a sample dataset with columns: rooms, bedrooms, population, households
    return np.array([
        [0, 0, 0, 6, 3, 900, 300],  # Sample 1
        [0, 0, 0, 8, 4, 1600, 400], # Sample 2
        [0, 0, 0, 10, 5, 2000, 500],# Sample 3
    ])


# Test for `add_bedrooms_per_room=True`
def test_transform_add_bedrooms_per_room(sample_data):
    # Arrange
    transformer = CombinedAttributesAdder(add_bedrooms_per_room=True)
    expected_output = np.c_[
        sample_data,
        # rooms_per_household
        sample_data[:, 3] / sample_data[:, 6],
        # population_per_household
        sample_data[:, 5] / sample_data[:, 6],
        # bedrooms_per_room
        sample_data[:, 4] / sample_data[:, 3],
    ]

    # Act
    transformed_data = transformer.transform(sample_data)

    # Assert
    assert np.allclose(transformed_data, expected_output), "Transformed data does not match expected output."


# Test for `add_bedrooms_per_room=False`
def test_transform_without_bedrooms_per_room(sample_data):
    # Arrange
    transformer = CombinedAttributesAdder(add_bedrooms_per_room=False)
    expected_output = np.c_[
        sample_data,
        # rooms_per_household
        sample_data[:, 3] / sample_data[:, 6],
        # population_per_household
        sample_data[:, 5] / sample_data[:, 6],
    ]

    # Act
    transformed_data = transformer.transform(sample_data)

    # Assert
    assert np.allclose(transformed_data, expected_output), "Transformed data does not match expected output when `add_bedrooms_per_room=False`."


# Test for edge cases: division by zero
# def test_transform_division_by_zero():
#     # Arrange
#     data_with_zero = np.array([
#         [0, 0, 0, 0, 0, 900, 0],  # households is 0
#     ])
#     transformer = CombinedAttributesAdder(add_bedrooms_per_room=True)

#     # Act
#     transformed_data = transformer.transform(data_with_zero)

#     # Assert
#     assert np.isnan(transformed_data).any() or np.isinf(transformed_data).any(), \
#         "Expected np.nan or np.inf in the output due to division by zero."
#     # # Act & Assert
#     # with pytest.raises(ZeroDivisionError):
#     #     transformer.transform(data_with_zero)


# Test for invalid input types
def test_transform_invalid_input():
    # Arrange
    invalid_data = [["not", "a", "number"]]
    transformer = CombinedAttributesAdder(add_bedrooms_per_room=True)

    # Act & Assert
    with pytest.raises(TypeError):
        transformer.transform(invalid_data)


# Test the `fit` method
def test_fit_method(sample_data):
    # Arrange
    transformer = CombinedAttributesAdder(add_bedrooms_per_room=True)

    # Act
    result = transformer.fit(sample_data)

    # Assert
    assert result is transformer, "`fit` method should return self."
