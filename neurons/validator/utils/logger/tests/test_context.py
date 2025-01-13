from unittest.mock import patch

import pytest

from neurons.validator.utils.logger.context import (
    get_context,
    logger_context,
    start_session,
    start_trace,
)


class TestLoggerContext:
    # Define a pytest fixture to mock uuid4
    @pytest.fixture
    def mock_uuid(self):
        with patch("neurons.validator.utils.logger.context.uuid4") as mock:
            yield mock

    def test_get_context_no_context(self):
        # Clear any existing context
        logger_context.set(None)

        # Get context before calling start_trace
        context = get_context()

        # Ensure the context is an empty dictionary
        assert context == {}

    def test_start_trace_creates_unique_session_ids(self, mock_uuid):
        session_ids = [
            "12345678-1234-5678-1234-567812345672",
            "87654321-4321-8765-4321-876543218765",
        ]
        # Mock uuid4 to return a fixed value based on the call count
        mock_uuid.side_effect = session_ids

        # Clear any existing context
        logger_context.set(None)

        # First session start
        start_session()
        first_context = get_context()
        first_trace_id = first_context["session_id"]

        # Second session start
        start_session()
        second_context = get_context()
        second_trace_id = second_context["session_id"]

        # Ensure the trace IDs are different and as mocked
        assert first_trace_id == session_ids[0]
        assert second_trace_id == session_ids[1]

        # Verify uuid4 was called twice (once for each session)
        assert mock_uuid.call_count == 2

    def test_start_trace_creates_unique_trace_ids(self, mock_uuid):
        trace_ids = [
            "12345678-1234-5678-1234-567812345672",
            "87654321-4321-8765-4321-876543218765",
        ]
        # Mock uuid4 to return a fixed value based on the call count
        mock_uuid.side_effect = trace_ids

        # Clear any existing context
        logger_context.set(None)

        # First trace start
        start_trace()
        first_context = get_context()
        first_trace_id = first_context["trace_id"]

        # Second trace start
        start_trace()
        second_context = get_context()
        second_trace_id = second_context["trace_id"]

        # Ensure the trace IDs are different and as mocked
        assert first_trace_id == trace_ids[0]
        assert second_trace_id == trace_ids[1]

        # Verify uuid4 was called twice (once for each trace)
        assert mock_uuid.call_count == 2
