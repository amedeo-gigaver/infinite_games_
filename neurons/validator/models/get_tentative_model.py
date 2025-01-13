"""
WARNING: use this chatGPT script manually and with careful review.
Quick and dirty script to generate Pydantic models from SQLite table schema.
"""

import sqlite3
import sys
from typing import Any

# Simple SQLite-to-Python type mapping, adjust as needed.
sqlite_to_python = {
    "TEXT": str,
    "INTEGER": int,
    "REAL": float,
    "BLOB": bytes,
    # If your schema uses DATETIME/TIMESTAMP text fields, handle them as str or datetime
    # depending on your requirements.
}


def convert_default(default_value: str | None) -> str | None:
    # Try to intelligently guess the Python type for the default
    if default_value is None:
        return None
    # If it's a numeric string, try converting to int or float
    try:
        if "." in default_value:
            float_val = float(default_value)
            return float_val
        else:
            int_val = int(default_value)
            return int_val
    except ValueError:
        pass
    # If it's something like CURRENT_TIMESTAMP, just leave it as string
    return default_value


def create_pydantic_model_from_table(db_path: str, table_name: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    conn.close()

    field_definitions = []
    indent = "    "  # 4 spaces for indentation inside the class

    for col in columns:
        _, col_name, col_type, col_notnull, col_default, _ = col
        col_type_upper = (col_type or "TEXT").upper()
        py_type = sqlite_to_python.get(col_type_upper, Any)

        is_optional = col_notnull == 0
        default = convert_default(col_default)

        if is_optional:
            annotation = f"Optional[{py_type.__name__}]" if py_type != Any else "Optional[Any]"
        else:
            annotation = py_type.__name__ if py_type != Any else "Any"

        if is_optional:
            if default is None:
                default_str = "= None"
            else:
                default_repr = repr(default)
                default_str = f"= Field(default={default_repr})"
        else:
            if default is None:
                default_str = ""
            else:
                default_repr = repr(default)
                default_str = f"= Field(default={default_repr})"

        field_definitions.append(f"{indent}{col_name}: {annotation} {default_str}".rstrip())

    model_name = f"{table_name.capitalize()}Model"

    model_code = [
        "from pydantic import BaseModel, Field",
        "from typing import Optional, Any",
        "",
        f"class {model_name}(BaseModel):",
    ]

    if field_definitions:
        model_code.extend(field_definitions)
    else:
        # If no fields, still add indentation for pass
        model_code.append("    pass")

    # Add the config line at the end, also indented
    model_code.append(f'{indent}model_config = {{"arbitrary_types_allowed": True}}')

    return "\n".join(model_code)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_model.py <db_path> <table_name>")
        sys.exit(1)

    db_path = sys.argv[1]
    table_name = sys.argv[2]

    print(
        """WARNING: This script does not handle all SQLite types and constraints."""
        """\nPlease review the generated Pydantic model and adjust as needed."""
        """\n\n\n"""
    )
    result = create_pydantic_model_from_table(db_path, table_name)
    print(result)
