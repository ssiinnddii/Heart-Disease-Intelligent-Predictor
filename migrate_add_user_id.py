"""Add missing user_id column to the predictions table."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "instance" / "predictions.db"

if not DB_PATH.exists():
    print("Database not found. It will be created fresh on next app start.")
    exit(0)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check if column already exists
cursor.execute("PRAGMA table_info(predictions)")
columns = [row[1] for row in cursor.fetchall()]

if "user_id" not in columns:
    cursor.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER REFERENCES users(id)")
    conn.commit()
    print("Successfully added user_id column to predictions table.")
else:
    print("user_id column already exists.")

conn.close()
