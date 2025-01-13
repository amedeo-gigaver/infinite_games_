import logging

from neurons.validator.utils.logger.context import start_session, start_trace
from neurons.validator.utils.logger.formatters import ConsoleFormatter, JSONFormatter


class InfiniteGamesLogger(logging.Logger):
    @property
    def start_session(self):
        return start_session

    @property
    def start_trace(self):
        return start_trace


logging.setLoggerClass(InfiniteGamesLogger)


# Override the default logging.Logger.makeRecord method to keep extra data
def make_record_with_extra(self, *args, **kwargs):
    record = original_makeRecord(self, *args, **kwargs)

    record._extra = args[-2]

    return record


# Replace the original makeRecord method with the custom implementation
original_makeRecord = logging.Logger.makeRecord
logging.Logger.makeRecord = make_record_with_extra


# Factory function to create and configure a logger with multiple handlers
def create_logger(
    name: str = None, level: any = logging.DEBUG, message_log: bool = False
) -> InfiniteGamesLogger:
    # Initialize the logger with the specified name and level
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Console message handler
    if message_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ConsoleFormatter())
        logger.addHandler(console_handler)

    # Console JSON handler
    json_handler = logging.StreamHandler()
    json_handler.setFormatter(JSONFormatter())
    logger.addHandler(json_handler)

    return logger


logger = create_logger("validator")
db_logger = create_logger("db")
