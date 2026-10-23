"""
=============================================================================
MODULE 3: Database & Storage
=============================================================================
Manages the local SQLite database for the Face Recognition system.
Handles:
  - Database initialization (schema creation)
  - User enrollment (register new faces)
  - Template retrieval (for matching)
  - User management (list, delete, update)
=============================================================================
"""

import sqlite3
import os
import datetime
import numpy as np
from typing import Optional, List, Tuple

from feature_extraction import template_to_bytes, bytes_to_template


# ---------------------------------------------------------------------------
# 3.1  Database Paths & Schema
# ---------------------------------------------------------------------------

# Default database file location (in project root)
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "face_recognition.db")

# SQL schema — single table for simplicity
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    enrolled_at TEXT    NOT NULL,
    template    BLOB    NOT NULL,
    template_len INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS enrollment_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    action      TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""


# ---------------------------------------------------------------------------
# 3.2  Database Connection Manager
# ---------------------------------------------------------------------------

class FaceDatabase:
    """
    SQLite-backed database for Face Recognition.

    Provides methods to:
        - Initialize the database schema
        - Enroll new users (store feature templates)
        - Retrieve all templates for matching
        - List, update, and delete users

    Uses context management (with statement) for safe connection handling.
    Each public method opens a new connection, executes, and closes cleanly.

    Attributes:
        db_path (str): Path to the SQLite database file.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize the FaceDatabase.

        Args:
            db_path: Path to the SQLite .db file. Created if it doesn't exist.
        """
        self.db_path = db_path
        self._initialize_db()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Open a new SQLite connection with WAL journal mode for better
        concurrent read performance.

        Returns:
            sqlite3.Connection object.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")   # Better read concurrency
        conn.execute("PRAGMA foreign_keys=ON;")    # Enforce FK constraints
        conn.row_factory = sqlite3.Row             # Row access by column name
        return conn

    def _initialize_db(self) -> None:
        """
        Create database tables if they don't already exist.
        Safe to call on every startup (uses CREATE TABLE IF NOT EXISTS).
        """
        conn = self._get_connection()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            print(f"[DB] Database initialized at: {self.db_path}")
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # 3.3  Enrollment — Register a New User
    # -----------------------------------------------------------------------

    def enroll_user(self, name: str, template: np.ndarray) -> int:
        """
        Enroll a new user by storing their name and feature template.

        The feature template (NumPy float32 array) is serialized to raw
        bytes and stored in a BLOB column. This avoids any lossy encoding
        and allows exact reconstruction.

        Args:
            name:     Full name of the user to enroll.
            template: 1D float32 feature vector from feature_extraction.py.

        Returns:
            user_id (int): Auto-assigned database ID for the new user.

        Raises:
            ValueError: If name is empty or template is invalid.
        """
        if not name or not name.strip():
            raise ValueError("User name cannot be empty.")
        if template is None or len(template) == 0:
            raise ValueError("Feature template cannot be empty.")

        name = name.strip()
        timestamp = datetime.datetime.now().isoformat()
        template_blob = template_to_bytes(template)

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO users (name, enrolled_at, template, template_len)
                VALUES (?, ?, ?, ?)
                """,
                (name, timestamp, template_blob, len(template))
            )
            user_id = cursor.lastrowid

            # Log enrollment action
            conn.execute(
                """
                INSERT INTO enrollment_log (user_id, action, timestamp)
                VALUES (?, ?, ?)
                """,
                (user_id, "ENROLLED", timestamp)
            )
            conn.commit()
            print(f"[DB] Enrolled user '{name}' with ID={user_id} "
                  f"(template_len={len(template)}).")
            return user_id

        finally:
            conn.close()

    def update_user_template(self, user_id: int,
                             new_template: np.ndarray) -> bool:
        """
        Update the feature template for an existing user.

        Useful for re-enrollment or template averaging over multiple captures.

        Args:
            user_id:      ID of the user to update.
            new_template: New 1D float32 feature vector.

        Returns:
            True if the update succeeded, False if user_id not found.
        """
        template_blob = template_to_bytes(new_template)
        timestamp = datetime.datetime.now().isoformat()

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                UPDATE users
                SET template=?, template_len=?
                WHERE id=?
                """,
                (template_blob, len(new_template), user_id)
            )
            if cursor.rowcount == 0:
                print(f"[DB] User ID={user_id} not found.")
                return False

            # Log update action
            conn.execute(
                """
                INSERT INTO enrollment_log (user_id, action, timestamp)
                VALUES (?, ?, ?)
                """,
                (user_id, "TEMPLATE_UPDATED", timestamp)
            )
            conn.commit()
            print(f"[DB] Updated template for user ID={user_id}.")
            return True

        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # 3.4  Template Retrieval — For Matching
    # -----------------------------------------------------------------------

    def get_all_users(self) -> List[Tuple[int, str, np.ndarray]]:
        """
        Retrieve all enrolled users with their feature templates.

        This is called by the Matching module to compare a query template
        against all stored templates.

        Returns:
            List of tuples: [(user_id, name, template_array), ...]
            where template_array is a float32 NumPy array.
            Returns empty list if no users are enrolled.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT id, name, template FROM users ORDER BY id"
            ).fetchall()

            users = []
            for row in rows:
                user_id = row["id"]
                name = row["name"]
                template = bytes_to_template(row["template"])
                users.append((user_id, name, template))

            return users

        finally:
            conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[Tuple[int, str, str, np.ndarray]]:
        """
        Retrieve a single user record by ID.

        Args:
            user_id: The user's database ID.

        Returns:
            Tuple (user_id, name, enrolled_at, template) or None.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT id, name, enrolled_at, template FROM users WHERE id=?",
                (user_id,)
            ).fetchone()

            if row is None:
                return None

            template = bytes_to_template(row["template"])
            return (row["id"], row["name"], row["enrolled_at"], template)

        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # 3.5  User Management
    # -----------------------------------------------------------------------

    def list_users(self) -> List[Tuple[int, str, str, int]]:
        """
        List all enrolled users (without templates — for display only).

        Returns:
            List of tuples: [(user_id, name, enrolled_at, template_len), ...]
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT id, name, enrolled_at, template_len FROM users ORDER BY id"
            ).fetchall()
            return [(r["id"], r["name"], r["enrolled_at"], r["template_len"])
                    for r in rows]
        finally:
            conn.close()

    def delete_user(self, user_id: int) -> bool:
        """
        Delete a user and all associated enrollment logs from the database.

        Args:
            user_id: The user's database ID.

        Returns:
            True if deletion succeeded, False if user not found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM users WHERE id=?", (user_id,))
            conn.commit()
            if cursor.rowcount == 0:
                print(f"[DB] User ID={user_id} not found.")
                return False
            print(f"[DB] Deleted user ID={user_id}.")
            return True
        finally:
            conn.close()

    def clear_all_users(self) -> int:
        """
        Delete ALL users from the database (use with caution!).

        Returns:
            Number of rows deleted.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM users")
            conn.execute("DELETE FROM enrollment_log")
            conn.commit()
            count = cursor.rowcount
            print(f"[DB] Cleared {count} users from database.")
            return count
        finally:
            conn.close()

    def get_enrollment_log(self, user_id: Optional[int] = None) -> List[sqlite3.Row]:
        """
        Retrieve the enrollment history log.

        Args:
            user_id: If provided, filter log for this specific user.
                     If None, return all log entries.

        Returns:
            List of Row objects with fields: id, user_id, action, timestamp.
        """
        conn = self._get_connection()
        try:
            if user_id is not None:
                rows = conn.execute(
                    "SELECT * FROM enrollment_log WHERE user_id=? ORDER BY timestamp",
                    (user_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM enrollment_log ORDER BY timestamp"
                ).fetchall()
            return list(rows)
        finally:
            conn.close()

    def __repr__(self) -> str:
        users = self.list_users()
        return (f"FaceDatabase(path='{self.db_path}', "
                f"num_users={len(users)})")
