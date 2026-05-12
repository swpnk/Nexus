from __future__ import annotations

import logging
from typing import Any, cast

import structlog
from structlog.typing import EventDict, WrappedLogger


def add_agent_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Attach standard agent fields to a structlog event when available."""
    agent = event_dict.pop("agent_instance", None)
    if agent is not None:
        event_dict["agent"] = agent.__class__.__name__
        event_dict["agent_id"] = agent.context.agent_id
        event_dict["state"] = agent.state.value
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog JSON logging with UTC timestamps and log level filtering."""
    logging.basicConfig(level=log_level.upper(), format="%(message)s")
    structlog.configure(
        processors=[
            add_agent_context,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_agent_logger(agent: Any) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to an agent instance."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger().bind(agent_instance=agent))
