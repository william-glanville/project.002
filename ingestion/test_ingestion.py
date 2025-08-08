import os
import json
import logging
import constants
from ingestor import Ingestor, IngestorConfig

# --- Config ---
SUBJECT_01_FOLDER = constants.DATA_RAW01_DIR
COLLECTION_NAME = "159.341"
CHUNK_SIZE = 180
OVERLAP = 40
RESET_DATASTORE = True

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger( str(constants.get_log_file("test_ingestor")) )

    # Step 1: Configure ingestion
    config = IngestorConfig(
        collection_name=COLLECTION_NAME,
        collection_path=SUBJECT_01_FOLDER,
        reset_datastore=RESET_DATASTORE
    )

    # Step 2: Initialize ingestor
    ingestor = Ingestor(config)

    # Step 3: Run ingestion
    logger.info("Starting test ingestion...")
    ingestor.ingest()
    logger.info("Ingestion complete.")

    # Step 4: Optional output summary
    output_path = os.path.join("output", f"{COLLECTION_NAME}_summary.json")
    os.makedirs("output", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "collection_name": COLLECTION_NAME,
            "source_path": str(SUBJECT_01_FOLDER),
            "reset": RESET_DATASTORE
        }, f, indent=2)

    logger.info(f"📄 Summary saved to {output_path}")


if __name__ == "__main__":
    main()
