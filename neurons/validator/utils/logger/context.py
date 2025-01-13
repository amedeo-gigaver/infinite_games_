from contextvars import ContextVar
from uuid import uuid4

logger_context = ContextVar("logger_context", default=None)


def start_session():
    context = logger_context.get() or {}

    logger_context.set({**context, "session_id": str(uuid4())})


def start_trace():
    context = logger_context.get() or {}

    logger_context.set({**context, "trace_id": str(uuid4())})


def get_context() -> dict:
    return logger_context.get() or {}
