import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY introuvable : vérifie le fichier .env à la racine du projet")