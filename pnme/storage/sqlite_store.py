import sqlite3
import numpy as np
import json
from datetime import datetime

class SQLiteStore:
    def __init__(self, db_path="pnme_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Table for semantic triples and their HDC representation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT,
                    relation TEXT,
                    object TEXT,
                    context TEXT,
                    timestamp TEXT,
                    vector BLOB
                )
            """)
            # Table for base symbols and their random HDC vectors
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    name TEXT PRIMARY KEY,
                    vector BLOB
                )
            """)
            conn.commit()

    def store_memory(self, subject, relation, obj, context, vector):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (subject, relation, object, context, timestamp, vector)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (subject, relation, obj, context, datetime.now().isoformat(), vector.tobytes()))
            conn.commit()

    def get_all_memories(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT subject, relation, object, context, timestamp, vector FROM memories")
            rows = cursor.fetchall()
            return [
                {
                    "subject": r[0],
                    "relation": r[1],
                    "object": r[2],
                    "context": r[3],
                    "timestamp": r[4],
                    "vector": np.frombuffer(r[5], dtype=np.int8)
                } for r in rows
            ]

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
            return {r[0]: np.frombuffer(r[1], dtype=np.int8) for r in rows}
