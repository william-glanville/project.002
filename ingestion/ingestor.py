import os
import uuid
from abc import ABC, abstractmethod

import chromadb
import fitz  # PyMuPDF
import mysql.connector
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

import constants

class MySQLSyncHandler:
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
            print(f"MySQL connection error: {err}")
            raise

    def insert_chunk_metadata(self, chunks):
        query = """
        INSERT INTO mcq_metadata (chunk_id, source_file, source_path, page_number, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            source_path = VALUES(source_path),
            page_number = VALUES(page_number)
        """

        for chunk in chunks:
            chunk_id = chunk["id"]
            meta = chunk["metadata"]
            values = (
                chunk_id,
                meta.get("source_file"),
                meta.get("source_path"),
                meta.get("page")
            )
            self.cursor.execute(query, values)

        self.conn.commit()

    def reset_chunk_metadata_table(self):
        drop_metadata_query = "DROP TABLE IF EXISTS mcq_metadata;"
        create_metadata_query = """
        CREATE TABLE mcq_metadata (
            chunk_id VARCHAR(255) PRIMARY KEY,
            source_path TEXT,
            page_number INT,
            topic_tags JSON,
            difficulty ENUM('easy', 'medium', 'hard'),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        self.cursor.execute(drop_metadata_query)
        self.cursor.execute(create_metadata_query)
        self.conn.commit()
        print("chunk_metadata table reset.")

    def close(self):
        self.cursor.close()
        self.conn.close()

class FileCrawler:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def get_files(self, postfix: str ):
        return [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith(postfix)]

    def get_pdf_files(self):
        return self.get_files(constants.FILE_TYPE_PDF)

    def get_docx_files(self):
        return self.get_files(constants.FILE_TYPE_DOCX)

    def get_pptx_files(self):
        return self.get_files(constants.FILE_TYPE_PPTX)

class PDFParser:
    def __init__(self):
        pass

    def extract_text(self, filepath):
        doc = fitz.open(filepath)
        all_pages = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                all_pages.append({"page": page_num, "text": text})
        return all_pages

class ChunkEmbedder:
    def __init__(self, model_name=constants.MODEL_EMBED_NAME, chunk_size=300, overlap=50):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, pages, source_file):
        chunks = []
        for page in pages:
            words = page["text"].split()
            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk_text = " ".join(words[i:i+self.chunk_size])
                ...
                chunk = {
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "metadata": {
                        "source_file": os.path.basename(source_file),
                        "page": page["page"],
                        "source_path": os.path.abspath(source_file)
                    }
                }
                chunks.append( chunk )

        return chunks

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
        self.settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=location,
            anonymized_telemetry=False
        )
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=constants.MODEL_EMBED_NAME)
        self.collection_name = collection_name
        self.client = chromadb.Client(settings=self.settings)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_functions=self.embed_fn,
        )

    def insert_chunks(self, chunks):
        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        self.collection.add(documents=texts, ids=ids, metadatas=metadatas)

    def reset_collection(self, collection_name="rag_chunks"):
        try:
            self.client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' dropped.")
        except Exception as e:
            print(f"Could not drop collection '{collection_name}': {e}")

        # Optional: reinitialize for insert readiness
        self.client.create_collection(collection_name)
        print(f"Collection '{collection_name}' recreated.")

    def reset_all_collections(self):
        for col in self.client.list_collections():
            self.client.delete_collection(col.name)
            print(f"Dropped collection '{col.name}'")

