"""
CLI script: wipe AETHERIS SQLite memory tables.

Usage:
    python scripts/reset_memory.py --all
    python scripts/reset_memory.py --user user123
    python scripts/reset_memory.py --checkpoints
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetheris.config import get_settings


def reset_long_term_memory(user_id: str | None, db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    if user_id:
        conn.execute("DELETE FROM user_memory WHERE user_id = ?", (user_id,))
        print(f"Cleared long-term memory for user '{user_id}'")
    else:
        conn.execute("DELETE FROM user_memory")
        print("Cleared ALL long-term memory")
    conn.commit()
    conn.close()


def reset_checkpoints(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        try:
            conn.execute(f"DELETE FROM {table}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()
    print("Cleared session checkpoints")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Reset AETHERIS memory stores")
    parser.add_argument("--all", action="store_true", help="Wipe both long-term memory and checkpoints")
    parser.add_argument("--user", type=str, help="Wipe long-term memory for a specific user")
    parser.add_argument("--checkpoints", action="store_true", help="Wipe session checkpoints only")
    args = parser.parse_args()

    if not any([args.all, args.user, args.checkpoints]):
        parser.print_help()
        sys.exit(1)

    if args.all or args.user:
        reset_long_term_memory(args.user, settings.sqlite_memory_path)

    if args.all or args.checkpoints:
        reset_checkpoints(settings.sqlite_checkpoints_path)


if __name__ == "__main__":
    main()
