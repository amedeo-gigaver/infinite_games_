import json
import logging
import sys

from neurons.validator.utils.git import commit_short_hash
from neurons.validator.utils.logger.context import logger_context
from neurons.validator.utils.logger.formatters import ConsoleFormatter, JSONFormatter
from neurons.validator.version import __version__


class TestConsoleFormatter:
    def test_console_formatter_info_log(self):
        formatter = ConsoleFormatter()

        record = logging.LogRecord(
            "test_logger", logging.INFO, "test_module", 1, "This is an info message", None, None
        )
        formatted_message = formatter.format(record)

        # Ensure the formatted message contains the log message
        assert "This is an info message" in formatted_message
        assert "INFO" in formatted_message
        assert "test_logger" in formatted_message


class TestJsonFormatter:
    def test_json_formatter_basic_log(self):
        formatter = JSONFormatter()

        # Create a log record with INFO level
        record = logging.LogRecord(
            "test_logger", logging.INFO, "test_module", 1, "This is a JSON log message", None, None
        )

        logger_context.set({"context_key": "context_value"})

        # Simulate the context
        formatted_message = formatter.format(record)

        log_data = json.loads(formatted_message)

        # Ensure the log data contains the correct fields
        assert "timestamp" in log_data
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "This is a JSON log message"
        assert log_data["context_key"] == "context_value"

    def test_json_formatter_with_extra_data(
        self,
    ):
        formatter = JSONFormatter()

        # Add some extra data to the record
        extra_data = {"key": "value"}

        record = logging.LogRecord(
            "test_logger",
            logging.INFO,
            "test_module",
            1,
            "This is a log with extra data",
            None,
            None,
        )
        record.__dict__["_extra"] = extra_data

        formatted_message = formatter.format(record)
        log_data = json.loads(formatted_message)

        # Ensure the extra data is included in the JSON log output
        assert "data" in log_data
        assert log_data["data"] == extra_data

    def test_json_formatter_with_exception(self):
        formatter = JSONFormatter()

        # Create an exception to log
        try:
            1 / 0
        except ZeroDivisionError:
            exc_info = sys.exc_info()

            record = logging.LogRecord(
                "test_logger",
                logging.ERROR,
                "test_module",
                1,
                "This is a log with an exception",
                None,
                exc_info,
            )

        formatted_message = formatter.format(record)
        log_data = json.loads(formatted_message)

        # Ensure the exception information is included in the JSON log output
        assert "exception" in log_data
        assert "ZeroDivisionError" in log_data["exception"]

    def test_json_formatter_with_version(
        self,
    ):
        formatter = JSONFormatter()

        record = logging.LogRecord(
            "test_logger",
            logging.INFO,
            "test_module",
            1,
            "This is a log",
            None,
            None,
        )

        formatted_message = formatter.format(record)
        log_data = json.loads(formatted_message)

        # Ensure the version data is included in the JSON log output
        assert "version" in log_data
        assert log_data["version"] == __version__

        assert "commit_hash" in log_data
        assert log_data["commit_hash"] == commit_short_hash
