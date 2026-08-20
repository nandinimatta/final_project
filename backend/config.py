import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# Database Configuration Settings
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://nandinimatta1202_db_user:ts0MJIrCsyx0K2rN@softpredictapp.psc1ife.mongodb.net/?appName=Softpredictapp"
)
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "softpredict_db")

# Path Configurations
DB_PATH = BASE_DIR / "medical_records.db"
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Default AI Correction Pipeline Settings
DEFAULT_CORRECTION_FLOW = os.getenv("DEFAULT_CORRECTION_FLOW", "gcn")
