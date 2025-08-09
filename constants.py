import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DIR_DATA = ROOT / "data"
DIR_LOGS = ROOT / "logs"

DIR_RETRIEVAL = ROOT / "retrieval"
DIR_RETRIEVAL_TEMPLATES = DIR_RETRIEVAL / "templates"
DIR_RETRIEVAL_STATIC = DIR_RETRIEVAL / "static"

DATA_RAW01_DIR = DIR_DATA / "159.341/"
DATA_CHROMA_DB_DIR = DIR_DATA / "chroma/"

FILE_TYPE_ANY = ".*"
FILE_TYPE_BASIC = r"\.(pdf|txt|md|docx)$"
FILE_TYPE_TEXT = r"\.(txt|md)$"
FILE_TYPE_PDF = r"\.(pdf)$"
FILE_TYPE_DOCX = r"\.(docx)$"
FILE_TYPE_PPTX = r"\.(pptx)$"

MYSQL_USER = "mcquser"
MYSQL_PASSWORD = "mcquser"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_SCHEMA = "mcq"

def get_log_file(name):
    return DIR_LOGS / f"{name}.log"

def get_data_file(name):
    return DIR_DATA / f"{name}"

def get_file_path(path, name):
    return f"{path}/{name}"

MODEL_EMBED_NAME = "all-MiniLM-L6-v2"
MODEL_CHUNK_TAGGER_NAME = "facebook/bart-large-mnli"


def safe_text(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

# Optional: whitelist of conceptual nouns to retain even if ambiguous
CONCEPT_WHITELIST = {
    "embedding", "pipeline", "workflow", "classifier", "metadata", "intent", "entity",
    "vector", "cache", "corpus", "token", "paraphrase", "hyperparameter", "model", "c", "C#", "cpp","c++","asm"
}

# Optional: blacklist of generic nouns to exclude
CONCEPT_BLACKLIST = {
    "file", "line", "project", "info", "data", "error", "use", "load", "main", "call", "stack"
}

CUSTOM_STOPWORDS = {
    "way", "there", "they", "them", "those", "thing", "stuff", "something", "anything",
    "everything", "nothing", "etc", "use", "used", "using", "get", "got", "make", "made"
}
