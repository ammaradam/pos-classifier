"""Environment-variable helpers shared across projects."""

import os


def env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}
