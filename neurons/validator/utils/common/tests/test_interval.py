from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from neurons.validator.utils.common.interval import (
    get_interval_iso_datetime,
    get_interval_start_minutes,
)


class TestInterval:
    @pytest.mark.parametrize(
        "test_time,expected",
        [
            # Test exact epoch start
            (datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 0),
            # Test multiple days later
            (datetime(2025, 1, 3, 4, 0, 0, tzinfo=timezone.utc), 530160),
            # Test with non-zero minutes and seconds (should round down)
            (datetime(2024, 1, 1, 4, 45, 30, tzinfo=timezone.utc), 240),
        ],
    )
    def test_get_interval_start_minutes(self, test_time, expected):
        with patch("neurons.validator.utils.common.interval.datetime") as mock_datetime:
            # Configure mock to return our test time when now() is called
            mock_datetime.now.return_value = test_time

            result = get_interval_start_minutes()

            assert result == expected

    def test_get_interval_iso_datetime(self):
        interval_start_minutes = 500000
        expected = "2024-12-13T05:20:00+00:00"

        result = get_interval_iso_datetime(interval_start_minutes)

        assert result == expected
