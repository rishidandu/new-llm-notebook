#!/usr/bin/env python3
"""
Build RAG System with Qdrant Cloud Cluster - TURBO VERSION
This script uses batch embedding and aggressive optimization for speed
"""

import sys
import os
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import asyncio
import aiohttp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config
from src.rag.embeddings import EmbeddingGenerator
from src.rag.qdrant_store import QdrantStore
from src.utils.rag_optimized_processor import RAGOptimizedProcessor
from src.utils.data_processor import DataProcessor, Document

def main():
    """Build RAG system with Qdrant cloud cluster using turbo optimization"""

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info("🚀 Building RAG system with Qdrant cloud cluster (TURBO)...")

    try:
        # Load configuration
        config = Config()
        logger.info("✅ Configuration loaded")

        # Initialize components
        embedding_generator = EmbeddingGenerator()
        qdrant_store = QdrantStore(config.COLLECTION_NAME)
        processor = RAGOptimizedProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

        logger.info("✅ Components initialized")

        # Get all data files
        data_files = get_all_data_files()
        logger.info(f"📊 Found {len(data_files)} data files to process")

        if not data_files:
            logger.warning("⚠️ No data files found. Creating sample data...")
            create_sample_data()
            data_files = get_all_data_files()

        # Process files with turbo optimization
        total_documents = process_files_turbo(data_files, embedding_generator, qdrant_store, processor)

        elapsed_time = time.time() - start_time
        logger.info(f"✅ RAG system built successfully!")
        logger.info(f"📊 Total documents processed: {total_documents}")
        logger.info(f"⏱️ Total time: {elapsed_time:.2f} seconds")
        logger.info(f"🚀 Documents per second: {total_documents/elapsed_time:.1f}")

        # Get final stats
        stats = qdrant_store.get_stats()
        logger.info(f"🔧 Final stats: {stats}")

        return True

    except Exception as e:
        logger.error(f"❌ Error building RAG system: {e}")
        return False

def get_all_data_files() -> List[Dict[str, Any]]:
    """Get all data files with their metadata"""
    files = []
    
    # Reddit files
    reddit_dir = Path("data/raw/reddit")
    if reddit_dir.exists():
        for file_path in reddit_dir.glob("*.jsonl"):
            files.append({
                'path': str(file_path),
                'type': 'reddit',
                'processor': 'rag_optimized'
            })
    
    # ASU web files
    asu_web_dir = Path("data/raw/asu_web")
    if asu_web_dir.exists():
        for file_path in asu_web_dir.glob("*.jsonl"):
            files.append({
                'path': str(file_path),
                'type': 'asu_web',
                'processor': 'standard'
            })
    
    # ASU grades files
    grades_dir = Path("data/raw/raw_asu_grades")
    if grades_dir.exists():
        for file_path in grades_dir.glob("*.csv"):
            files.append({
                'path': str(file_path),
                'type': 'asu_grades',
                'processor': 'grades'
            })
    
    return files

def process_single_file_turbo(file_info: Dict[str, Any], embedding_generator: EmbeddingGenerator, 
                             qdrant_store: QdrantStore, processor: RAGOptimizedProcessor) -> int:
    """Process a single file with turbo optimization"""
    file_path = file_info['path']
    file_type = file_info['type']
    processor_type = file_info['processor']
    
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 Processing {file_type} file: {os.path.basename(file_path)}")
    
    try:
        documents = []
        
        if processor_type == 'rag_optimized':
            # Use RAG-optimized processing for Reddit data
            documents = list(processor.process_reddit_data_rag_optimized(file_path))
        elif processor_type == 'standard':
            # Use standard processing for ASU web data
            from src.utils.data_processor import DataProcessor
            standard_processor = DataProcessor(chunk_size=1000, chunk_overlap=200)
            documents = list(standard_processor.process_asu_data(file_path))
        elif processor_type == 'grades':
            # Use grades processing for ASU grades data
            from src.utils.asu_grades_processor import ASUGradesProcessor
            grades_processor = ASUGradesProcessor(chunk_size=1000, chunk_overlap=200)
            documents = list(grades_processor.process_grades_data(file_path))
        
        if not documents:
            logger.warning(f"⚠️ No documents generated from {file_path}")
            return 0
        
        # TURBO: Generate embeddings in large batches
        batch_size = 100  # Much larger batches
        embeddings = []
        enhanced_documents = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_texts = [doc.content for doc in batch_docs]
            
            # Generate embeddings for entire batch at once
            batch_embeddings = embedding_generator.get_embeddings_batch(batch_texts)
            embeddings.extend(batch_embeddings)
            
            # Process documents
            for j, doc in enumerate(batch_docs):
                if processor_type == 'rag_optimized':
                    enhanced_metadata = {
                        **doc.metadata,
                        'quality_score': doc.quality_score,
                        'conversation_context': doc.conversation_context
                    }
                    enhanced_doc = Document(
                        id=doc.id,
                        content=doc.content,
                        metadata=enhanced_metadata,
                        source=doc.source
                    )
                    enhanced_documents.append(enhanced_doc)
                else:
                    enhanced_documents.append(doc)
        
        # Add to Qdrant in large batches
        qdrant_store.add_documents(enhanced_documents, embeddings)
        
        logger.info(f"   ✅ Added {len(documents)} documents from {os.path.basename(file_path)}")
        return len(documents)
        
    except Exception as e:
        logger.error(f"❌ Error processing {file_path}: {e}")
        return 0

def process_files_turbo(data_files: List[Dict[str, Any]], embedding_generator: EmbeddingGenerator,
                       qdrant_store: QdrantStore, processor: RAGOptimizedProcessor) -> int:
    """Process all files with turbo optimization"""
    logger = logging.getLogger(__name__)
    total_documents = 0
    
    # Use more workers for turbo processing
    max_workers = min(100, len(data_files))  # Increased to 100 workers for optimal performance
    
    logger.info(f"🔄 Processing {len(data_files)} files with {max_workers} turbo workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file_turbo, file_info, embedding_generator, qdrant_store, processor): file_info
            for file_info in data_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file_info = future_to_file[future]
            try:
                doc_count = future.result()
                total_documents += doc_count
                logger.info(f"📊 Progress: {total_documents} total documents processed")
            except Exception as e:
                logger.error(f"❌ Error processing {file_info['path']}: {e}")
    
    return total_documents

def create_sample_data():
    """Create sample data for testing"""
    import json

    # Create sample Reddit data
    sample_reddit_data = [
        {
            "id": "sample_1",
            "title": "Best study spots on ASU campus",
            "text": "The Hayden Library is one of the best study spots on ASU campus. It has multiple floors with quiet study areas, group study rooms, and 24/7 access during finals week. The Noble Library is also great for engineering students. The Memorial Union has good spots for group study sessions.",
            "url": "https://reddit.com/r/ASU/comments/sample1",
            "source": "reddit",
            "ingested_at": "2025-07-26T00:00:00Z",
            "metadata": {
                "subreddit": "ASU",
                "score": 45,
                "num_comments": 12,
                "author": "asu_student",
                "post_type": "submission"
            }
        },
        {
            "id": "sample_2", 
            "title": "Cool places to hang out around ASU",
            "text": "Mill Avenue is the main entertainment district near ASU with restaurants, bars, and shops. The Tempe Town Lake is great for outdoor activities. The ASU Art Museum is free for students. The Sun Devil Stadium area is perfect for game days and events.",
            "url": "https://reddit.com/r/ASU/comments/sample2",
            "source": "reddit",
            "ingested_at": "2025-07-26T00:00:00Z",
            "metadata": {
                "subreddit": "ASU",
                "score": 32,
                "num_comments": 8,
                "author": "campus_explorer",
                "post_type": "submission"
            }
        }
    ]

    # Ensure directory exists
    os.makedirs("data/raw/reddit", exist_ok=True)

    # Write sample data
    with open("data/raw/reddit/sample_data.jsonl", "w") as f:
        for item in sample_reddit_data:
            f.write(json.dumps(item) + "\n")

    print("✅ Created sample data for testing")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 