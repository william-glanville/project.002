import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DIR_DATA = ROOT / "data"
DIR_LOGGS = ROOT / "logs"

DATA_RAW01_DIR = DIR_DATA / "raw.01/"
DATA_CHROMA_DB_DIR = DIR_DATA / "chroma/"

FILE_TYPE_ANY = ".*"
FILE_TYPE_BASIC = r"\.(pdf|txt|md|docx)$"
FILE_TYPE_TEXT = r"\.(txt|md)$"
FILE_TYPE_PDF = r"\.(pdf)$"
FILE_TYPE_DOCX = r"\.(docx)$"
FILE_TYPE_PPTX = r"\.(pptx)$"

def get_log_file(name):
    return DIR_LOGGS / f"{name}.log"

def get_data_file(name):
    return DIR_DATA / f"{name}"

MODEL_EMBED_NAME = "all-MiniLM-L6-v2"
MODEL_CHUNK_TAGGER_NAME = "facebook/bart-large-mnli"


def safe_text(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
