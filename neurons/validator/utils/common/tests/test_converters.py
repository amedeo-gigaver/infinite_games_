from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import BaseModel

from neurons.validator.utils.common.converters import (
    pydantic_models_to_dataframe,
    torch_or_numpy_to_int,
)


class ExperimentalModel(BaseModel):
    id: int
    name: str
    score: float
    is_active: bool
    created_at: datetime
    optional_field: Optional[int] = None


class TestPydanticModelToDataFrame:
    def test_empty_list(self):
        # Passing an empty list should return an empty DataFrame
        df = pydantic_models_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_single_model(self):
        # Convert a single model
        model = ExperimentalModel(
            id=1,
            name="Alice",
            score=9.5,
            is_active=True,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            optional_field=None,
        )
        df = pydantic_models_to_dataframe([model])
        assert len(df) == 1
        assert df.loc[0, "id"] == 1
        assert df.loc[0, "name"] == "Alice"
        assert df.loc[0, "score"] == 9.5
        assert df.loc[0, "is_active"] == True  # noqa
        assert df.loc[0, "optional_field"] is pd.NA
        # Check that created_at is a datetime64 type
        assert pd.api.types.is_datetime64_ns_dtype(df["created_at"])
        # Check other dtypes
        assert pd.api.types.is_integer_dtype(df["id"]) or pd.api.types.is_integer_dtype(
            df["id"].dropna()
        )  # nullable integer
        assert pd.api.types.is_float_dtype(df["score"])
        # For optional_field with None, after conversion it might be a nullable integer type (Int64) or float depending on pandas version
        # We can at least check that it's one of the nullable dtypes
        assert str(df["optional_field"].dtype) in ("Int64", "float64", "object")

    def test_multiple_models(self):
        models = [
            ExperimentalModel(
                id=1,
                name="Alice",
                score=9.5,
                is_active=True,
                created_at=datetime(2024, 1, 1, 12, 0),
                optional_field=10,
            ),
            ExperimentalModel(
                id=2,
                name="Bob",
                score=7.2,
                is_active=False,
                created_at=datetime(2024, 1, 2, 15, 30),
                optional_field=None,
            ),
        ]

        df = pydantic_models_to_dataframe(models)
        assert len(df) == 2
        assert list(df["id"]) == [1, 2]
        assert list(df["name"]) == ["Alice", "Bob"]
        assert list(df["score"]) == [9.5, 7.2]
        assert list(df["is_active"]) == [True, False]

        # Check types
        assert pd.api.types.is_datetime64_ns_dtype(df["created_at"])
        assert pd.api.types.is_float_dtype(df["score"])
        # optional_field: first row is int, second is None => should be a nullable integer or float
        assert str(df["optional_field"].dtype) in ("Int64", "float64", "object")

    def test_unmapped_type(self):
        # If the model has a field type that isn't in the type_mapping, it should still work without error
        class UnmappedModel(BaseModel):
            id: int
            custom_field: complex  # complex is not in our type mapping
            created_at: datetime

        model = UnmappedModel(id=3, custom_field=3 + 4j, created_at=datetime(2024, 1, 1))
        df = pydantic_models_to_dataframe([model])
        assert len(df) == 1
        # The custom_field should be included but might be object dtype
        assert "custom_field" in df.columns
        assert df.loc[0, "custom_field"] == 3 + 4j
        # dtype might be object since no mapping was done
        assert df["custom_field"].dtype in (object, np.dtype("complex128"))


class TestTorchOrNumpyToInteger:
    def test_torch_tensor_single_value(self):
        tensor = torch.tensor([42.0])
        result = torch_or_numpy_to_int(tensor)
        assert result == 42

    def test_torch_parameter_single_value(self):
        param = torch.nn.Parameter(torch.tensor([42.0]))
        result = torch_or_numpy_to_int(param)
        assert result == 42

    def test_numpy_array_single_value(self):
        ndarray = np.array([42])
        result = torch_or_numpy_to_int(ndarray)
        assert result == 42

    def test_numpy_int64(self):
        value = np.int64(42)
        result = torch_or_numpy_to_int(value)
        assert result == 42

    def test_torch_tensor_zero_dimension(self):
        tensor = torch.tensor(42)
        result = torch_or_numpy_to_int(tensor)
        assert result == 42

    def test_numpy_array_zero_dimension(self):
        ndarray = np.array(42)  # 0-D array
        result = torch_or_numpy_to_int(ndarray)
        assert result == 42

    def test_torch_parameter_zero_dimension(self):
        param = torch.nn.Parameter(torch.tensor(42.0))  # 0-D tensor
        result = torch_or_numpy_to_int(param)
        assert result == 42

    def test_torch_tensor_multiple_values(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(RuntimeError):
            torch_or_numpy_to_int(tensor)

    def test_torch_parameter_multiple_values(self):
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        with pytest.raises(RuntimeError):
            torch_or_numpy_to_int(param)

    def test_numpy_array_multiple_values(self):
        ndarray = np.array([1, 2, 3])
        with pytest.raises(
            ValueError, match="NDArray contains multiple elements; cannot convert to int."
        ):
            torch_or_numpy_to_int(ndarray)

    def test_unsupported_type(self):
        value = "not_supported"
        with pytest.raises(TypeError, match="Unsupported type for conversion to int."):
            torch_or_numpy_to_int(value)
