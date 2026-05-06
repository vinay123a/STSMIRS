"""
KMD — Key Management Database
Off-chain SQLite store for encrypted tourist identity records.
Schema matches report Table 3.2.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional


class RecordStatus(Enum):
    ACTIVE = "ACTIVE"
    SCHEDULED_FOR_DELETION = "SCHEDULED_FOR_DELETION"
    DELETED = "DELETED"


class KMD:
    """Key Management Database backed by SQLite."""

    DDL = """
    CREATE TABLE IF NOT EXISTS identities (
        adm_ref              TEXT PRIMARY KEY,
        encrypted_identity   BLOB NOT NULL,
        wrapped_key          BLOB NOT NULL,
        cipher_meta          TEXT NOT NULL,
        deletion_timestamp   TEXT NOT NULL,
        status               TEXT NOT NULL DEFAULT 'ACTIVE'
    );
    """

    def __init__(self, db_path: str = "kmd.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(self.DDL)
        self.conn.commit()

    # ── Store ──────────────────────────────────────────────────

    def store_identity(
        self,
        encrypted_identity: bytes,
        wrapped_key: bytes,
        cipher_meta: dict,
        retention_hours: int = 48,
    ) -> str:
        """
        Store an encrypted tourist identity record.
        Returns the generated adm_ref (UUID string).
        """
        adm_ref = str(uuid.uuid4())
        deletion_ts = datetime.now(timezone.utc) + timedelta(hours=retention_hours)

        self.conn.execute(
            """
            INSERT INTO identities
                (adm_ref, encrypted_identity, wrapped_key, cipher_meta,
                 deletion_timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                adm_ref,
                encrypted_identity,
                wrapped_key,
                json.dumps(cipher_meta),
                deletion_ts.isoformat(),
                RecordStatus.ACTIVE.value,
            ),
        )
        self.conn.commit()
        return adm_ref

    # ── Retrieve ───────────────────────────────────────────────

    def get_identity(self, adm_ref: str) -> Optional[dict]:
        """Fetch a record by adm_ref.  Returns None if not found."""
        row = self.conn.execute(
            "SELECT * FROM identities WHERE adm_ref = ?", (adm_ref,)
        ).fetchone()
        if row is None:
            return None
        return {
            "adm_ref": row["adm_ref"],
            "encrypted_identity": row["encrypted_identity"],
            "wrapped_key": row["wrapped_key"],
            "cipher_meta": json.loads(row["cipher_meta"]),
            "deletion_timestamp": row["deletion_timestamp"],
            "status": row["status"],
        }

    # ── Key Expiry (NFR3 — 48-hour privacy guarantee) ─────────

    def schedule_deletion(self, adm_ref: str) -> None:
        """Mark a record for deletion (intermediate state)."""
        self.conn.execute(
            "UPDATE identities SET status = ? WHERE adm_ref = ?",
            (RecordStatus.SCHEDULED_FOR_DELETION.value, adm_ref),
        )
        self.conn.commit()

    def run_key_expiry_sweep(self) -> int:
        """
        Zero-out wrapped_key for all records past their deletion_timestamp.
        Returns the number of records deleted.
        """
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            """
            UPDATE identities
               SET wrapped_key = X'',
                   status      = ?
             WHERE status     != ?
               AND deletion_timestamp <= ?
            """,
            (RecordStatus.DELETED.value, RecordStatus.DELETED.value, now),
        )
        self.conn.commit()
        return cursor.rowcount

    # ── Housekeeping ──────────────────────────────────────────

    def close(self):
        self.conn.close()
