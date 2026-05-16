"""
Structured Logging Configuration.
Uses the structlog library to provide both human-readable console output
and machine-readable JSON logs.
"""
import logging
import structlog

def setup_logger(json_format: bool = False):
    """
    Configures the global logger with a standard pipeline of processors.
    If json_format is True, output will be structured JSON (ideal for the daemon).
    """
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Default to pretty console output
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str):
    """Returns a bound logger for a specific module."""
    return structlog.get_logger(name)
