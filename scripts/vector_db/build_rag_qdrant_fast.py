#!/usr/bin/env python3
"""
Fast Qdrant RAG Build Script for Render
Optimized for Render's build environment with limited time and resources
"""

import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.qdrant_store import QdrantStore
from src.rag.embeddings import EmbeddingGenerator
from src.utils.rag_optimized_processor import RAGOptimizedProcessor
from src.utils.data_processor import DataProcessor, Document
from config.settings import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_file_fast(file_path: str, processor: RAGOptimizedProcessor, embedding_generator: EmbeddingGenerator, qdrant_store: QdrantStore) -> int:
    """Process a single file with fast batching"""
    try:
        logger.info(f"🔄 Processing: {os.path.basename(file_path)}")
        
        # Process documents
        if 'reddit' in file_path:
            documents = list(processor.process_reddit_data_rag_optimized(file_path))
        else:
            # Use standard processor for ASU data
            standard_processor = DataProcessor(chunk_size=1000, chunk_overlap=200)
            documents = list(standard_processor.process_asu_data(file_path))
        
        if not documents:
            logger.info(f"   ⚠️  No documents found in {os.path.basename(file_path)}")
            return 0
        
        logger.info(f"   📄 Generated {len(documents)} documents")
        
        # Process in small batches for Render
        batch_size = 50  # Small batches for Render
        total_added = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Generate embeddings
            embeddings = []
            enhanced_documents = []
            
            for doc in batch:
                try:
                    embedding = embedding_generator.get_embedding(doc.content)
                    embeddings.append(embedding)
                    
                    # Enhance metadata
                    enhanced_metadata = {
                        **doc.metadata,
                        'quality_score': getattr(doc, 'quality_score', 0.5),
                        'conversation_context': getattr(doc, 'conversation_context', '')
                    }
                    
                    enhanced_doc = Document(
                        id=doc.id,
                        content=doc.content,
                        metadata=enhanced_metadata,
                        source=doc.source
                    )
                    enhanced_documents.append(enhanced_doc)
                    
                except Exception as e:
                    logger.warning(f"   ⚠️  Error processing document: {e}")
                    continue
            
            if enhanced_documents:
                try:
                    qdrant_store.add_documents(enhanced_documents, embeddings)
                    total_added += len(enhanced_documents)
                    logger.info(f"   ✅ Added batch {i//batch_size + 1}: {len(enhanced_documents)} documents")
                except Exception as e:
                    logger.error(f"   ❌ Error adding batch to Qdrant: {e}")
        
        return total_added
        
    except Exception as e:
        logger.error(f"❌ Error processing {file_path}: {e}")
        return 0

def main():
    """Main function for fast Qdrant build"""
    start_time = time.time()
    logger.info("🚀 Starting Fast Qdrant RAG Build for Render")
    
    # Initialize components
    config = Config()
    processor = RAGOptimizedProcessor(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    embedding_generator = EmbeddingGenerator()
    
    # Initialize Qdrant store
    qdrant_store = QdrantStore(
        collection_name=config.COLLECTION_NAME,
        host=config.QDRANT_HOST,
        api_key=config.QDRANT_API_KEY
    )
    
    # Get data files
    data_files = []
    
    # Add Reddit files
    if os.path.exists(config.REDDIT_RAW_DIR):
        for filename in os.listdir(config.REDDIT_RAW_DIR):
            if filename.endswith('.jsonl'):
                data_files.append(os.path.join(config.REDDIT_RAW_DIR, filename))
    
    # Add ASU files
    if os.path.exists(config.ASU_RAW_DIR):
        for filename in os.listdir(config.ASU_RAW_DIR):
            if filename.endswith('.jsonl'):
                data_files.append(os.path.join(config.ASU_RAW_DIR, filename))
    
    # Add ASU grades files
    if os.path.exists(config.ASU_GRADES_RAW_DIR):
        for filename in os.listdir(config.ASU_GRADES_RAW_DIR):
            if filename.endswith('.csv'):
                data_files.append(os.path.join(config.ASU_GRADES_RAW_DIR, filename))
    
    logger.info(f"📁 Found {len(data_files)} data files")
    
    if not data_files:
        logger.warning("⚠️  No data files found! Creating sample data...")
        # Create minimal sample data for testing
        sample_docs = [
            Document(
                id="sample_1",
                content="ASU is a great university with excellent professors.",
                metadata={"source": "sample", "type": "general"},
                source="sample"
            ),
            Document(
                id="sample_2", 
                content="Computer Science at ASU offers many courses and has good faculty.",
                metadata={"source": "sample", "type": "academic"},
                source="sample"
            )
        ]
        
        embeddings = [embedding_generator.get_embedding(doc.content) for doc in sample_docs]
        qdrant_store.add_documents(sample_docs, embeddings)
        logger.info("✅ Added sample data to Qdrant")
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Fast Qdrant build completed in {elapsed_time:.2f} seconds")
        return
    
    # Process files with limited parallelism for Render
    max_workers = min(5, len(data_files))  # Limited workers for Render
    total_documents = 0
    
    logger.info(f"🔄 Processing with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_file_fast, file_path, processor, embedding_generator, qdrant_store): file_path
            for file_path in data_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                documents_added = future.result()
                total_documents += documents_added
                logger.info(f"✅ Completed {os.path.basename(file_path)}: +{documents_added} documents")
            except Exception as e:
                logger.error(f"❌ Error processing {os.path.basename(file_path)}: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ Fast Qdrant build completed!")
    logger.info(f"📊 Total documents processed: {total_documents}")
    logger.info(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    logger.info(f"🚀 Documents per second: {total_documents/elapsed_time:.1f}")

if __name__ == "__main__":
    main() 