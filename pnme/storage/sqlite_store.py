import sqlite3
import numpy as np
import json
from datetime import datetime
from typing import List
from ..core.schema import MemoryRecord

class SQLiteStore:
    def __init__(self, db_path="pnme_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency and durability
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Version Tracking
            cursor.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            
            # Initial Tables (V1)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    subject TEXT,
                    relation TEXT,
                    object TEXT,
                    memory_type TEXT,
                    source TEXT,
                    session_id TEXT,
                    agent_id TEXT,
                    context TEXT,
                    timestamp_created TEXT,
                    timestamp_last_accessed TEXT,
                    confidence REAL,
                    strength REAL,
                    reinforcement_count INTEGER,
                    decay_factor REAL,
                    provenance TEXT,
                    tags TEXT,
                    vector_encoding_version TEXT,
                    symbolic_version TEXT,
                    privacy_level INTEGER,
                    vector BLOB
                )
            """)
            # Symbols Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    name TEXT PRIMARY KEY,
                    vector BLOB
                )
            """)
            
            # Audit & Event Logging
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    event_type TEXT,
                    timestamp TEXT,
                    details TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_log (
                    access_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    timestamp TEXT,
                    query_type TEXT
                )
            """)
            
            # Soft Deletes (Tombstones)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tombstones (
                    memory_id TEXT PRIMARY KEY,
                    timestamp_deleted TEXT
                )
            """)
            
            # Engine Settings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_relation ON memories(relation)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_object ON memories(object)")
            
            # Ensure version is set
            cursor.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('version', '2')")
            conn.commit()

    def maintenance(self):
        """Perform database optimization."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            conn.commit()

    def store_memory_record(self, record: MemoryRecord):
        storage_data = record.to_storage_dict()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            keys = ", ".join(storage_data.keys())
            placeholders = ", ".join(["?" for _ in storage_data])
            cursor.execute(f"""
                INSERT OR REPLACE INTO memories ({keys})
                VALUES ({placeholders})
            """, tuple(storage_data.values()))
            conn.commit()

    def get_all_records(self) -> List[MemoryRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories")
            rows = cursor.fetchall()
            return [MemoryRecord.from_row(r) for r in rows]

    def store_symbol(self, name, vector):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO symbols (name, vector) VALUES (?, ?)", (name, vector.tobytes()))
            conn.commit()

    def load_symbols(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, vector FROM symbols")
            rows = cursor.fetchall()
            return {r['name']: np.frombuffer(r['vector'], dtype=np.int8) for r in rows}

    def update_memory_metadata(self, memory_id: str, updates: dict):
        """Efficiently update specific metadata fields."""
        if not updates:
            return
        fields = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values())
        values.append(memory_id)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE memories SET {fields} WHERE memory_id = ?", values)
            
            # Log event if strength changed significantly or reinforcement happened
            if "reinforcement_count" in updates or "strength" in updates:
                cursor.execute("""
                    INSERT INTO memory_events (memory_id, event_type, timestamp, details)
                    VALUES (?, ?, ?, ?)
                """, (memory_id, "REINFORCE", datetime.now().isoformat(), json.dumps(updates)))
            
            conn.commit()

    def log_access(self, memory_id: str, query_type: str = "direct"):
        """Log a memory access event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO access_log (memory_id, timestamp, query_type)
                VALUES (?, ?, ?)
            """, (memory_id, datetime.now().isoformat(), query_type))
            conn.commit()

    def export_jsonl(self, export_path: str):
        """Export all memories to a JSONL file (vectors as lists)."""
        records = self.get_all_records()
        with open(export_path, 'w', encoding='utf-8') as f:
            for rec in records:
                d = rec.to_dict()
                # Vector to list for JSON
                d['vector'] = rec.vector.tolist()
                f.write(json.dumps(d) + "\n")

    def import_jsonl(self, import_path: str):
        """Import memories from a JSONL file."""
        with open(import_path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                # Convert list back to vector
                vec = np.array(d['vector'], dtype=np.int8)
                del d['vector']
                
                # Check for existing memory_id to avoid accidental overwrite if not intended?
                # Actually, store_memory_record uses INSERT OR REPLACE
                record = MemoryRecord(vector=vec, **d)
                self.store_memory_record(record)

    def set_setting(self, key: str, value: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
            conn.commit()

    def get_setting(self, key: str, default=None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row['value'] if row else default
