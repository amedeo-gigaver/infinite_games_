from shutil import disk_usage
from sqlite3 import sqlite_version
from sys import version_info


def tuple_version_to_str(version: tuple) -> str:
    return ".".join(map(str, version))


def assert_requirements():
    # Assert Python version
    system_python_version = version_info[:3]

    MIN_PYTHON_VERSION = (3, 10, 0)

    if system_python_version < MIN_PYTHON_VERSION:
        raise AssertionError(
            (
                f"Python version must be at least {tuple_version_to_str(MIN_PYTHON_VERSION)}"
                f", your version is {tuple_version_to_str(system_python_version)}"
            )
        )

    # Assert SQLite version
    MIN_SQLITE_VERSION = (3, 37, 0)

    system_sqlite_version = tuple(map(int, sqlite_version.split(".")))

    # Major has to be the same
    if system_sqlite_version[0] != MIN_SQLITE_VERSION[0]:
        raise AssertionError(
            (
                f"SQLite major version must be {MIN_SQLITE_VERSION[0]}"
                f", your version is {system_sqlite_version[0]}"
            )
        )

    if system_sqlite_version < MIN_SQLITE_VERSION:
        raise AssertionError(
            (
                f"SQLite version must be at least {tuple_version_to_str(MIN_SQLITE_VERSION)}"
                f", your version is {tuple_version_to_str(system_sqlite_version)}"
            )
        )

    # Assert disk space
    # Get disk usage statistics for the specified folder
    _, __, free = disk_usage(".")

    MIN_DISK_FREE_SPACE = 5 * 1024 * 1024 * 1024  # In bytes

    if free < MIN_DISK_FREE_SPACE:
        raise AssertionError(
            (
                f"Disk space must be at least {MIN_DISK_FREE_SPACE / (1024 * 1024 * 1024)} GB"
                f", your free space is {round(free / (1024 * 1024 * 1024),2)} GB"
            )
        )

    return {
        "python_version": tuple_version_to_str(system_python_version),
        "sqlite_version": tuple_version_to_str(system_sqlite_version),
    }
