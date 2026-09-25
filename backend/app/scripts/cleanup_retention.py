import argparse

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services.retention_service import cleanup_expired_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    database = SessionLocal()
    try: print(cleanup_expired_data(database, get_settings(), dry_run=arguments.dry_run))
    finally: database.close()
