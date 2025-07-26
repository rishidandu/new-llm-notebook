#!/usr/bin/env python3
"""
Builds the ChromaDB vector store from raw data sources using parallel processing.

This script is optimized for local development and provides a fast and efficient
way to create a comprehensive RAG database from ASU, Reddit, and grades data.

Key Features:
- Parallel Processing: Uses a ThreadPoolExecutor to process multiple files concurrently.
- Batch Embedding: Generates embeddings in batches for improved efficiency.
- Optimized Data Processing: Leverages optimized RAG processors for high-quality output.
- Comprehensive Data Sources: Includes ASU web data, Reddit conversations, and ASU grades.
"""

import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.vector_store import VectorStore
from src.rag.embeddings import EmbeddingGenerator
from src.utils.rag_optimized_processor import RAGOptimizedProcessor
from src.utils.data_processor import DataProcessor, Document
from src.utils.asu_grades_processor import ASUGradesProcessor
from config.settings import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_file_batch(file_path: str, config: Config, processor: RAGOptimizedProcessor, grades_processor: ASUGradesProcessor) -> list[Document]:
    """Processes a single file and returns a list of documents."""
    try:
        logger.info(f"Processing file: {os.path.basename(file_path)}")
        
        if 'reddit' in file_path:
            documents = list(processor.process_reddit_data_rag_optimized(file_path))
        elif 'grades' in file_path:
            documents = list(grades_processor.process_grades_data(file_path))
        else:
            standard_processor = DataProcessor(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            documents = list(standard_processor.process_asu_data(file_path))
            
        logger.info(f"Generated {len(documents)} documents from {os.path.basename(file_path)}")
        return documents
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []

def main():
    """Main function to build the ChromaDB vector store."""
    start_time = time.time()
    logger.info("Starting ChromaDB build process...")

    config = Config()
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore(config.COLLECTION_NAME, config.VECTOR_DB_DIR)
    processor = RAGOptimizedProcessor(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    grades_processor = ASUGradesProcessor(config)
    
    # Get all data files
    data_files = []
    for dir_path in [config.REDDIT_RAW_DIR, config.ASU_RAW_DIR, config.ASU_GRADES_RAW_DIR]:
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(('.jsonl', '.csv')):
                    data_files.append(os.path.join(dir_path, filename))

    if not data_files:
        logger.warning("No data files found. Please run scrapers first.")
        return

    logger.info(f"Found {len(data_files)} data files to process.")

    all_documents = []
    # Use 10 workers since this is a local build
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(process_file_batch, fp, config, processor, grades_processor): fp for fp in data_files}
        for future in as_completed(future_to_file):
            all_documents.extend(future.result())

    logger.info(f"Total documents generated: {len(all_documents)}")
    
    if all_documents:
        # Generate embeddings in batches
        contents = [doc.content for doc in all_documents]
        embeddings = embedding_generator.get_embeddings(contents)
        
        # Add to vector store
        vector_store.add_documents(all_documents, embeddings)

    elapsed_time = time.time() - start_time
    logger.info(f"ChromaDB build completed in {elapsed_time:.2f} seconds.")
    logger.info(f"Total documents in store: {vector_store.get_stats()['total_documents']}")

if __name__ == "__main__":
    main() 