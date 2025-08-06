import os
import constants

from ingestor import FileCrawler, PDFParser, ChunkEmbedder, ChromaDBHandler, MySQLDBHandler

# --- Config ---
PDF_FOLDER = constants.DATA_RAW01_DIR
COLLECTION_NAME = "159.341"
CHUNK_SIZE = 180
OVERLAP = 40


def main():
    # Step 1: Crawl PDF Files
    crawler = FileCrawler(PDF_FOLDER)
    pdf_files = crawler.get_pdf_files()
    print(f"Found {len(pdf_files)} PDF(s):", pdf_files)

    # Step 2: Initialize Components
    parser = PDFParser()
    embedder = ChunkEmbedder(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    vector_db = ChromaDBHandler(location=constants.DATA_CHROMA_DB_DIR, collection_name=COLLECTION_NAME)
    mysql_sync = MySQLDBHandler()

    # step 3: Initialize MySQL/ChromasDB
    mysql_sync.reset_chunk_metadata_table()
    vector_db.reset_all_collections()

    all_chunks = []

    # Step 4: Process Each PDF
    for filepath in pdf_files:
        print(f"\nProcessing: {filepath}")
        pages = parser.extract_text(filepath)
        print(f"  → Pages extracted: {len(pages)}")

        chunks = embedder.chunk_text(collection_name=COLLECTION_NAME, pages=pages, source_file=filepath)
        print(f"  → Chunks created: {len(chunks)}")

        chunks = embedder.generate_embeddings(chunks)
        print("  → Embeddings generated")

        vector_db.insert_chunks(chunks)
        print("  → Chunks inserted into ChromaDB")

        mysql_sync.insert_chunk_metadata(chunks)
        print("  → Metadata synced to MySQL")

        all_chunks.extend(chunks)

    # Optional: Print summary
    print(f"\nTotal chunks processed: {len(all_chunks)}")

    # Cleanup
    mysql_sync.close()

if __name__ == "__main__":
    main()