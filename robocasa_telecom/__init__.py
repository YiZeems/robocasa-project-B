"""RoboCasa Telecom package organized in a robosuite-style layout."""

import logging
import warnings

# Patch gym_notices in-memory before gym is imported anywhere so the
# deprecation print() in gym/__init__.py never fires.
try:
    import gym_notices.notices as _gn  # noqa: F401
    _gn.notices = {}
except Exception:
    pass

# Suppress known irrelevant third-party warnings at import time.
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*upgrade to Gymnasium.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", message=".*mimicgen.*")


def _silence_third_party_loggers() -> None:
    """Set ERROR level on robosuite/robocasa loggers and strip their handlers.

    robosuite installs its own StreamHandler at WARNING level before we can
    intercept it; removing those handlers is the only reliable way to stop
    the mink and controller-config INFO messages from leaking to stdout.
    """
    for name in ("robosuite", "robocasa"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.handlers.clear()
        logger.propagate = False


_silence_third_party_loggers()

__all__ = [
    "envs",
    "rl",
    "tools",
    "utils",
]
