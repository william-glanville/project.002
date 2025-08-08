import os
import re
import uuid
import chromadb
import mysql.connector
import constants
import logging
import traceback
from abc import ABC, abstractmethod
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document
from semantic_filter import SemanticFilter
from reference_builder import ReferenceBuilder, SubjectExtractor, LocalLLMClient, ReferenceCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(constants.get_log_file("ingestor")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FileParser(ABC):
    @abstractmethod
    def extract_chunks(self, filepath):
        pass

class PDFParser(FileParser):
    def __init__(self):
        pass

    def extract_chunks(self, filepath):
        reader = PdfReader(filepath)
        chunks = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            chunks.append({
                "text": text,
                "page": i + 1,
                "source_path": filepath,
                "source_file": filepath.split("/")[-1]
            })
        return chunks

class DocxParser(FileParser):
    def __init__(self):
        pass

    def extract_chunks(self, filepath):
        doc = Document(filepath)
        chunks = []
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                chunks.append({
                    "text": para.text,
                    "page": i + 1,
                    "source_path": filepath,
                    "source_file": filepath.split("/")[-1]
                })
        return chunks

class TxtParser(FileParser):
    def __init__(self):
        pass

    def extract_chunks(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        chunks = []
        for i, line in enumerate(lines):
            if line.strip():
                chunks.append({
                    "text": line.strip(),
                    "page": i + 1,
                    "source_path": filepath,
                    "source_file": filepath.split("/")[-1]
                })
        return chunks

class FileParserFactory(FileParser):
    def __init__(self):
        self.parser_map = {
            ".pdf": PDFParser,
            ".docx": DocxParser,
            ".txt": TxtParser
        }

    def extract_chunks(self, filepath):
        parser = self._get_parser(filepath)
        return parser.extract_chunks(filepath)

    def _get_parser(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        parser_cls = self.parser_map.get(ext)

        if not parser_cls:
            raise ValueError(f"Unsupported file type: {ext}")

        return parser_cls()


class IngestorConfig:
    def __init__(
        self,
        collection_name,
        collection_path,
        file_filter_pattern=constants.FILE_TYPE_BASIC,
        embedder_model_name=constants.MODEL_EMBED_NAME,
        chunk_size=180,
        chunk_overlap=40,
        chromadb_path=constants.DATA_CHROMA_DB_DIR,
        mysql_user="mcquser",
        mysql_password="mcquser",
        mysql_host="localhost",
        mysql_port=3306,
        mysql_schema="mcq",
        reset_datastore=False
    ):
        self.collection_name = collection_name
        self.collection_path = collection_path
        self.file_filter_pattern = file_filter_pattern
        self.embedder_model_name = embedder_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chromadb_path = chromadb_path
        self.mysql_user = mysql_user
        self.mysql_password = mysql_password
        self.mysql_host = mysql_host
        self.mysql_port = mysql_port
        self.mysql_schema = mysql_schema
        self.reset_datastore = reset_datastore

class Ingestor:
    def __init__(self, config: IngestorConfig):
        self.config = config
        self.filter = SemanticFilter(logger=logging, model_name=config.embedder_model_name, threshold=0.3)
        self.llm = LocalLLMClient()
        self.cache = ReferenceCache( persistent=False, path=constants.get_data_file("reference.cache"))
        self.reference_builder = ReferenceBuilder( llm_client=self.llm, cache=self.cache )
        self.subject_extractor = SubjectExtractor( llm_client=self.llm )
        self.crawler = FileCrawler(config.collection_path, config.file_filter_pattern)
        self.parser_factory = FileParserFactory()
        self.embedder = ChunkEmbedder(
            model_name=config.embedder_model_name,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap
        )
        self.chromadb = ChromaDBHandler(config.chromadb_path, config.collection_name)
        self.mysql = MySQLDBHandler(
            user=config.mysql_user,
            password=config.mysql_password,
            host=config.mysql_host,
            port=config.mysql_port,
            database=config.mysql_schema
        )

        if config.reset_datastore:
            logger.info("Resetting ChromaDB and MySQL datastore...")
            self.chromadb.reset_collection(collection_name=config.collection_name)
            self.mysql.reset()

    def ingest(self):
        logger.info(f"Starting ingestion for collection: {self.config.collection_name}")
        file_paths = self.crawler.crawl()
        logger.info(f"Found {len(file_paths)} files matching pattern: {self.config.file_filter_pattern}")

        all_chunks = []

        for path in file_paths:
            try:
                logger.info(f"Parsing file: {path}")
                raw_chunks = self.parser_factory.extract_chunks(path)
                chunked = self.embedder.chunk_text(raw_chunks)
                subject_context = self.subject_extractor.extract_subject(chunked)
                # Build reference set from chunked data
                reference_texts = self.reference_builder.build_reference_set(chunks=chunked, context=subject_context)
                filtered = self.filter.filter_chunks(chunked, reference_texts)
                embedded = self.embedder.generate_embeddings(filtered)
                all_chunks.extend(embedded)
                logger.info(f"Processed {len(embedded)} chunks from {path}")
                logger.info(f"Generated {len(reference_texts)} reference sentences from extracted keywords")
                logger.info(f"Filtered down to {len(filtered)} chunks after semantic filtering")
            except Exception as e:
                logger.warning(f"Failed to process file {path}: {str(e)}")
                traceback.print_exc()
                continue

        logger.info(f"Storing {len(all_chunks)} chunks in ChromaDB and MySQL...")
        self.chromadb.insert_chunks(all_chunks)
        self.mysql.insert_chunk_metadata(all_chunks)
        logger.info("Ingestion complete.")


class FileCrawler:
    def __init__(self, root_dir, pattern=constants.FILE_TYPE_BASIC, recursive=False):
        """
        Args:
            root_dir (str): Directory to search
            pattern (str): Regex pattern to match file extensions
            recursive (bool): Whether to search subdirectories
        """
        self.root_dir = root_dir
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.recursive = recursive

    def crawl(self):
        matched_files = []

        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if self.pattern.search(filename):
                    full_path = os.path.join(dirpath, filename)
                    matched_files.append(full_path)

            if not self.recursive:
                break  # Only process the top-level directory

        return matched_files

class ChunkEmbedder:
    def __init__(self, model_name=constants.MODEL_EMBED_NAME, chunk_size=180, overlap=40):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, raw_chunks):
        processed_chunks = []

        for raw in raw_chunks:
            words = raw["text"].split()
            if len(words) <= self.chunk_size:
                # No need to split
                processed_chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": raw["text"],
                    "metadata": {
                        "source_file": raw["source_file"],
                        "source_path": raw["source_path"],
                        "page": raw["page"]
                    }
                })
            else:
                # Apply sliding window chunking
                for i in range(0, len(words), self.chunk_size - self.overlap):
                    chunk_text = " ".join(words[i:i + self.chunk_size])
                    processed_chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": chunk_text,
                        "metadata": {
                            "source_file": raw["source_file"],
                            "source_path": raw["source_path"],
                            "page": raw["page"]
                        }
                    })

        return processed_chunks

    def generate_embeddings(self, chunks):
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts)
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
        return chunks


class VectorDBHandler(ABC):
    @abstractmethod
    def insert_chunks(self, chunks):
        pass

    @abstractmethod
    def reset_collection(self, collection_name: str):
        pass

    @abstractmethod
    def reset_all_collections(self):
        pass



class ChromaDBHandler(VectorDBHandler):
    def __init__(self, location=constants.DATA_CHROMA_DB_DIR, collection_name="rag_chunks"):
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=constants.MODEL_EMBED_NAME)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=location)
        self.collection = None
        self.reset_collection(collection_name=collection_name)

    def insert_chunks(self, chunks):
        try:
            self.collection.add(
                documents=[chunk["text"] for chunk in chunks],
                ids=[chunk["id"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks]
            )
        except chromadb.errors.NotFoundError:
            # Recreate and retry
            logger.info(f"Collection '{self.collection_name}' not found. Reinitializing...")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embed_fn
            )
            self.insert_chunks(chunks)

    def reset_collection(self, collection_name="rag_chunks"):
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' dropped.")
        except Exception as e:
            logger.info(f"Could not drop collection '{collection_name}': {e}")

        # Optional: reinitialize for insert readiness
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embed_fn
        )
        logger.info(f"Collection '{collection_name}' recreated.")

    def reset_all_collections(self):
        for col in self.client.list_collections():
            self.client.delete_collection(col.name)
            logger.info(f"Dropped collection '{col.name}'")


class MySQLDBHandler:
    def __init__(self, host='localhost', port=3306, user='mcquser', password='mcquser', database='mcq'):
        try:
            self.conn = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            self.cursor = self.conn.cursor()
        except mysql.connector.Error as err:
            logger.info(f"MySQL connection error: {err}")
            raise

    def insert_chunk_metadata(self, chunks):
        query = """
            INSERT INTO mcq_metadata (chunk_id, source_file, source_path, page_number, collection_name, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW()) ON DUPLICATE KEY 
            UPDATE source_path =  VALUES (source_path), 
                   page_number =  VALUES (page_number), 
                   collection_name =  VALUES (collection_name)
        """

        for chunk in chunks:
            chunk_id = chunk["id"]
            meta = chunk["metadata"]
            values = (
                chunk_id,
                meta.get("source_file"),
                meta.get("source_path"),
                meta.get("page"),
                meta.get("collection_name")
            )

            self.cursor.execute(query, values)

        self.conn.commit()

    def reset(self):
        drop_questions_query = "DROP TABLE IF EXISTS mcq_questions;"
        drop_metadata_query = "DROP TABLE IF EXISTS mcq_metadata;"

        create_questions_table = """
            CREATE TABLE mcq_questions (    
                question_id VARCHAR(255) PRIMARY KEY,
                chunk_id VARCHAR(255),
                stem TEXT,
                options JSON,
                FOREIGN KEY(chunk_id) REFERENCES mcq_metadata(chunk_id)
            );
            """

        create_metadata_table = """
        CREATE TABLE mcq_metadata (
            chunk_id VARCHAR(255) PRIMARY KEY,
            source_file VARCHAR(255),
            source_path TEXT,
            page_number INT,
            topic_tags JSON,
            difficulty ENUM('easy', 'medium', 'hard'),
            collection_name VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        self.cursor.execute(drop_questions_query)
        self.cursor.execute(drop_metadata_query)

        self.cursor.execute(create_metadata_table)
        self.cursor.execute(create_questions_table)

        self.conn.commit()
        logger.info("Chunk metadata table reset.")

    def close(self):
        self.cursor.close()
        self.conn.close()
