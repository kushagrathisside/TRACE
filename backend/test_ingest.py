import logging

logging.basicConfig(level=logging.INFO)
from ingestion.ingestor import run_ingestion  # noqa: E402

print("Starting ingestion...")
run_ingestion()
print("Ingestion complete.")
