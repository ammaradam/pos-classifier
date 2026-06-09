"""SQLite write helpers shared across projects."""

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_with_retry(
    db_path: Path,
    sql: str,
    params: list[Any] | tuple[Any, ...] | list[tuple[Any, ...]],
    *,
    many: bool = False,
    max_retries: int = 3,
) -> None:
    """Execute an INSERT/UPDATE with exponential-backoff retry on lock contention."""
    for attempt in range(max_retries):
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            if many:
                conn.executemany(sql, params)  # type: ignore[arg-type]
            else:
                conn.execute(sql, params)  # type: ignore[arg-type]
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if conn:
                conn.close()
            if attempt < max_retries - 1:
                wait = 0.1 * (2**attempt)
                logger.warning(
                    "DB write retry %d/%d after %.2fs (locked: %s)",
                    attempt + 1,
                    max_retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            else:
                logger.error("DB write failed after %d retries: %s", max_retries, e)
                raise
        except Exception as e:
            if conn:
                conn.close()
            logger.error("Unexpected DB error: %s", e)
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
