from unittest.mock import patch

import pytest

from neurons.validator.utils.env import assert_requirements, tuple_version_to_str


class TestEnv:
    def test_tuple_version_to_str(self):
        assert tuple_version_to_str((3, 10, 0)) == "3.10.0"
        assert tuple_version_to_str((3, 7, 2)) == "3.7.2"
        assert tuple_version_to_str((2, 0, 0)) == "2.0.0"

    @pytest.mark.parametrize(
        "python_version,should_pass",
        [
            ((3, 10, 0), True),  # Minimum version - should pass
            ((3, 11, 0), True),  # Higher version - should pass
            ((3, 9, 9), False),  # Too low version - should fail
            ((2, 7, 0), False),  # Much too low - should fail
        ],
    )
    def test_python_version_check(self, python_version, should_pass):
        with patch("neurons.validator.utils.env.version_info", python_version):
            if should_pass:
                result = assert_requirements()
                assert result["python_version"] == tuple_version_to_str(python_version)
            else:
                with pytest.raises(AssertionError) as exc_info:
                    assert_requirements()
                assert "Python version must be at least" in str(exc_info.value)

    @pytest.mark.parametrize(
        "sqlite_version,should_pass",
        [
            ("3.37.0", True),  # Minimum version - should pass
            ("3.39.0", True),  # Higher version - should pass
            ("3.36.0", False),  # Too low version - should fail
            ("4.0.0", False),  # Wrong major version - should fail
            ("2.0.0", False),  # Wrong major version - should fail
        ],
    )
    def test_sqlite_version_check(self, sqlite_version: str, should_pass: bool):
        with patch("neurons.validator.utils.env.sqlite_version", sqlite_version):
            if should_pass:
                result = assert_requirements()
                assert result["sqlite_version"] == sqlite_version
            else:
                with pytest.raises(AssertionError) as exc_info:
                    assert_requirements()
                assert any(
                    x in str(exc_info.value)
                    for x in ["SQLite version must be at least", "SQLite major version must be"]
                )

    @pytest.mark.parametrize(
        "free_space,should_pass",
        [
            (6 * 1024 * 1024 * 1024, True),  # 6GB - should pass
            (5 * 1024 * 1024 * 1024, True),  # 5GB - should pass
            (4 * 1024 * 1024 * 1024, False),  # 4GB - should fail
            (1 * 1024 * 1024 * 1024, False),  # 1GB - should fail
        ],
    )
    def test_disk_space_check(self, free_space: int, should_pass: bool):
        mock_usage = (0, 0, free_space)  # total, used, free
        with patch(
            "neurons.validator.utils.env.disk_usage", return_value=mock_usage
        ) as disk_usage_mock:
            if should_pass:
                assert_requirements()
            else:
                with pytest.raises(AssertionError) as exc_info:
                    assert_requirements()
                assert "Disk space must be at least" in str(exc_info.value)

            disk_usage_mock.assert_called_once_with(".")

    def test_successful_requirements_check(self):
        # Test when all requirements are met
        version_info = (3, 10, 0)
        sqlite_version = "3.37.0"
        free_space = 6 * 1024 * 1024 * 1024

        with (
            patch("neurons.validator.utils.env.version_info", version_info),
            patch("neurons.validator.utils.env.sqlite_version", sqlite_version),
            patch("neurons.validator.utils.env.disk_usage", return_value=(0, 0, free_space)),
        ):
            result = assert_requirements()

            assert result == {"python_version": "3.10.0", "sqlite_version": "3.37.0"}
