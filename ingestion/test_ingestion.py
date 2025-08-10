import os
import json
import logging
import constants
from ingestor import Ingestor, IngestorConfig

# --- Config ---
CHUNK_SIZE = 180
OVERLAP = 40
RESET_DATASTORE = True

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger( str(constants.get_log_file("test_ingestor")) )

    for candidate in constants.DATA_FOR_INGESTION:
        collection_name = candidate["collection_name"]
        subject_folder = candidate["subject_folder"]
        # Step 1: Configure ingestion
        config = IngestorConfig(
            collection_name=collection_name,
            collection_path=subject_folder,
            reset_datastore=RESET_DATASTORE
        )

        # Step 2: Initialize ingestor
        ingestor = Ingestor(config)

        # Step 3: Run ingestion
        logger.info("Starting test ingestion...")
        ingestor.ingest()
        logger.info("Ingestion complete.")

        # Step 4: Optional output summary
        output_path = os.path.join("output", f"{collection_name}_summary.json")
        os.makedirs("output", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "collection_name": collection_name,
                "source_path": str(subject_folder),
                "reset": RESET_DATASTORE
            }, f, indent=2)

    logger.info(f"📄 Summary saved to {constants.safe_text(output_path)}")


if __name__ == "__main__":
    main()
