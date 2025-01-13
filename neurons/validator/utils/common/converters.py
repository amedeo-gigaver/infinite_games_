from typing import List, Union

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel


def torch_or_numpy_to_int(
    value: Union[torch.Tensor, torch.nn.Parameter, np.ndarray, np.int64]
) -> int:
    if isinstance(value, torch.Tensor):
        # Convert the Tensor to a scalar
        return int(value.item())

    elif isinstance(value, torch.nn.Parameter):
        # Convert the Parameter to a scalar if it contains a single value
        return int(value.data.item())

    elif isinstance(value, np.ndarray):
        # Convert ndarray to a scalar if it contains a single value
        if value.size == 1:  # Ensure it's a single element
            return int(value.item())
        else:
            raise ValueError("NDArray contains multiple elements; cannot convert to int.")

    elif isinstance(value, np.int64):
        # Convert Numpy value to int
        return value.item()

    else:
        raise TypeError("Unsupported type for conversion to int.")


def pydantic_models_to_dataframe(models: List[BaseModel]) -> pd.DataFrame:
    if not models:
        return pd.DataFrame()

    # Extract the field schema from the first model
    field_info = models[0].model_fields

    records = [m.model_dump() for m in models]

    df = pd.DataFrame(records)

    # Build a mapping from Python type to pandas/numpy dtype
    # This is a basic mapping and can be expanded as needed.
    type_mapping = {
        int: "Int64",  # pandas nullable integer type
        float: "float64",
        str: "string",  # pandas nullable string type
        bool: "boolean",  # pandas nullable boolean type
        # For datetime fields, pandas usually autodetects dtype='datetime64[ns]'
        # You can add custom handling if needed.
    }

    # Try to set the appropriate column dtypes based on annotations
    for field_name, field_def in field_info.items():
        if field_name in df.columns:
            # field_def.annotation holds the Python type
            py_type = field_def.annotation
            # If the annotation is something like Optional[int], extract the main type
            # In Pydantic v2, annotation might be a typing.Union or similar. Handle gracefully:
            origin = getattr(py_type, "__origin__", None)
            if origin is not None and origin is not str and hasattr(py_type, "__args__"):
                # Take the first arg as a guess (e.g., Union[int, None])
                py_type = py_type.__args__[0]

            # Map the Python type to a pandas dtype if possible
            mapped_dtype = type_mapping.get(py_type, None)
            if mapped_dtype is not None:
                # Convert the column
                df[field_name] = df[field_name].astype(mapped_dtype, errors="ignore")

    return df
