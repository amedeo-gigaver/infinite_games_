import json
import logging

from colorama import Back, Fore, Style

from neurons.validator.utils.git import commit_short_hash
from neurons.validator.utils.logger.context import get_context
from neurons.validator.version import __version__


# A custom logging formatter for console output with color coding for log levels
class ConsoleFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Back.WHITE,
    }

    LOG_FORMAT = (
        "$STD%(asctime)s.%(msecs)03d$RESET | "  # Timestamp with milliseconds
        "$LEVEL%(levelname)s$RESET | "  # Log level (DEBUG, INFO, etc.)
        "$STD%(name)s$RESET - "  # Logger name
        "$MESSAGE%(message)s$RESET"  # Log message itself
    )

    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        # Define the string format for the log message with placeholders for various parts

        # Define the date format for timestamp

        # Initialize the Formatter with the defined string format and date format
        super().__init__(fmt=self.LOG_FORMAT, datefmt=self.DATE_FORMAT)

    def format(self, record):
        """
        Custom format method to replace placeholders with actual values
        and add color coding based on the log level.
        """

        # Select the color for the log level
        level_color = self.COLORS.get(record.levelname, Fore.WHITE)

        # Get the log message
        record.message = record.getMessage()

        # Format the timestamp
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        # Format the final log message by replacing placeholders with values
        message = self.formatMessage(record)

        # Replace placeholders with actual color codes
        message = message.replace("$RESET", Style.RESET_ALL)
        message = message.replace("$STD", Fore.BLUE)
        message = message.replace("$LEVEL", level_color + Style.BRIGHT)
        message = message.replace("$MESSAGE", Fore.BLACK + Style.BRIGHT)

        return message


# A custom logging formatter that outputs logs in JSON format
class JSONFormatter(logging.Formatter):
    def format(self, record):
        """
        Format the log record into a JSON object, including log data and context.
        """
        # Retrieve any contextual data (e.g., trace ID) from the current logger context
        context = get_context()

        # Create a dictionary for the log record
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),  # Timestamp of the log event
            "level": record.levelname,  # Log level (INFO, ERROR, etc.)
            "logger": record.name,  # Logger name
            "message": record.getMessage(),  # The actual log message
            **context,  # Add context information
            "version": __version__,  # Version of the validator
            "commit_hash": commit_short_hash,  # Commit hash of the validator
        }

        # Add extra information to the log
        extra_info = record.__dict__.get("_extra", None)

        if extra_info:
            log_record["data"] = extra_info

        # If there is exception information, format it and add to the log record
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, indent=2)
