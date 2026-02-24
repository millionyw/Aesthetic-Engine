import os
import sqlite3
from datetime import datetime
from datetime import timezone

DB_PATH = "./data/gallery.db"


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                path TEXT PRIMARY KEY,
                pred_score REAL,
                human_score INTEGER,
                timestamp DATETIME
            )
            """
        )
        conn.commit()


def upsert_image(path: str, pred_score: float):
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO images (path, pred_score, human_score, timestamp)
            VALUES (?, ?, NULL, ?)
            ON CONFLICT(path) DO UPDATE SET
                pred_score = excluded.pred_score,
                timestamp = excluded.timestamp
            """,
            (path, float(pred_score), now),
        )
        conn.commit()


def update_human_score(path: str, human_score: int):
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO images (path, pred_score, human_score, timestamp)
            VALUES (?, NULL, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                human_score = excluded.human_score,
                timestamp = excluded.timestamp
            """,
            (path, int(human_score), now),
        )
        conn.commit()


def fetch_images():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT path, pred_score, human_score, timestamp
            FROM images
            ORDER BY COALESCE(human_score, pred_score) DESC, timestamp DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]
