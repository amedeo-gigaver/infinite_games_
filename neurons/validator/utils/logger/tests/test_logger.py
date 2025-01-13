import logging

import pytest

from neurons.validator.utils.logger.context import start_session, start_trace
from neurons.validator.utils.logger.formatters import ConsoleFormatter, JSONFormatter
from neurons.validator.utils.logger.logger import create_logger


class TestLogger:
    @pytest.fixture
    def logger(self):
        """Fixture to create a logger instance for testing."""
        return create_logger(name="test_logger", level=logging.CRITICAL)

    def test_logger_name_and_level(self, logger):
        """Test that the logger is configured with the correct name and level."""
        assert logger.name == "test_logger"
        assert logger.level == logging.CRITICAL

    def test_logger_handlers(self, logger):
        """Test that the logger has the correct handlers attached."""
        handlers = logger.handlers

        assert len(handlers) == 1  # Expecting 1 handler ( JSON)

        # Validate handler
        assert isinstance(handlers[0].formatter, JSONFormatter)

    def test_logger_with_message_handlers(self):
        """Test that the logger has the correct handlers attached."""
        logger = create_logger(name="test_logger", level=logging.CRITICAL, message_log=True)

        handlers = logger.handlers

        assert len(handlers) == 2  # Expecting 2 handlers (console and JSON)

        # Validate the type of each handler
        assert any(isinstance(handler.formatter, ConsoleFormatter) for handler in handlers)
        assert any(isinstance(handler.formatter, JSONFormatter) for handler in handlers)

    def test_logger_context_methods(self, logger):
        """Test that the logger has the correct context methods."""

        # Validate the type of each context method
        assert logger.start_session == start_session
        assert logger.start_trace == start_trace
