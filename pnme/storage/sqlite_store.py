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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    name TEXT PRIMARY KEY,
                    vector BLOB
                )
            """)
            
            # Ensure version is set
            cursor.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('version', '1')")
            conn.commit()

    def maintenance(self):
        """Perform database optimization."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            conn.commit()

    def store_memory_record(self, record: MemoryRecord):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO memories (
                    memory_id, subject, relation, object, memory_type, source, 
                    session_id, agent_id, context, timestamp_created, 
                    timestamp_last_accessed, confidence, strength, 
                    reinforcement_count, decay_factor, provenance, tags, 
                    vector_encoding_version, symbolic_version, privacy_level, vector
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.memory_id, record.subject, record.relation, record.object,
                record.memory_type, record.source, record.session_id, record.agent_id,
                record.context, record.timestamp_created, record.timestamp_last_accessed,
                record.confidence, record.strength, record.reinforcement_count,
                record.decay_factor, record.provenance, json.dumps(record.tags),
                record.vector_encoding_version, record.symbolic_version, 
                record.privacy_level, record.vector.tobytes()
            ))
            conn.commit()

    def get_all_records(self) -> List[MemoryRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories")
            rows = cursor.fetchall()
            records = []
            for r in rows:
                record = MemoryRecord(
                    memory_id=r['memory_id'],
                    subject=r['subject'],
                    relation=r['relation'],
                    object=r['object'],
                    memory_type=r['memory_type'],
                    source=r['source'],
                    session_id=r['session_id'],
                    agent_id=r['agent_id'],
                    context=r['context'],
                    timestamp_created=r['timestamp_created'],
                    timestamp_last_accessed=r['timestamp_last_accessed'],
                    confidence=r['confidence'],
                    strength=r['strength'],
                    reinforcement_count=r['reinforcement_count'],
                    decay_factor=r['decay_factor'],
                    provenance=r['provenance'],
                    tags=json.loads(r['tags']),
                    vector_encoding_version=r['vector_encoding_version'],
                    symbolic_version=r['symbolic_version'],
                    privacy_level=r['privacy_level'],
                    vector=np.frombuffer(r['vector'], dtype=np.int8)
                )
                records.append(record)
            return records

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
            conn.commit()
