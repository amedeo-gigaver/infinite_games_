import subprocess
from unittest.mock import patch

from neurons.validator.utils.git import get_commit_short_hash


class TestGit:
    def test_get_commit_short_hash(self):
        """Test successful retrieval of git commit hash"""

        with patch("subprocess.check_output") as mock_check_output:
            mock_hash = "mocked_hash"
            mock_check_output.return_value = mock_hash.encode("utf-8")

            result = get_commit_short_hash()

            # Verify the correct hash is returned
            assert result == mock_hash

            # Verify git command was called with correct arguments
            mock_check_output.assert_called_once_with(["git", "rev-parse", "--short", "HEAD"])

    def test_get_commit_short_hash_error(self):
        """Test error retrieving of git commit hash"""

        with patch("subprocess.check_output") as mock_check_output:
            mock_check_output.side_effect = subprocess.CalledProcessError(returncode=1, cmd="")

            result = get_commit_short_hash()

            # Verify empty hash is returned
            assert result == ""

            # Verify git command was called with correct arguments
            mock_check_output.assert_called_once_with(["git", "rev-parse", "--short", "HEAD"])
