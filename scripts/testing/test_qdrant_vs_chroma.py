#!/usr/bin/env python3
"""
Test script to compare ChromaDB vs Qdrant performance
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config
from src.rag.vector_store import VectorStore as ChromaVectorStore
from src.rag.qdrant_store import QdrantStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_search_performance(vector_store, store_name, test_queries):
    """Test search performance for a vector store"""
    logger.info(f"🔍 Testing {store_name} search performance...")
    
    # Get stats
    stats = vector_store.get_stats()
    total_docs = stats.get('total_documents', 0)
    logger.info(f"📊 {store_name} has {total_docs} documents")
    
    if total_docs == 0:
        logger.warning(f"⚠️ {store_name} has no documents to test")
        return
    
    # Test search performance
    search_times = []
    
    for i, query in enumerate(test_queries):
        logger.info(f"Testing query {i+1}: '{query}'")
        
        # Generate a simple test embedding (in real use, you'd use OpenAI)
        test_embedding = [0.1] * 1536  # Simple test embedding
        
        start_time = time.time()
        results = vector_store.search(test_embedding, top_k=5)
        end_time = time.time()
        
        search_time = end_time - start_time
        search_times.append(search_time)
        
        logger.info(f"  ⏱️ Search time: {search_time:.3f}s")
        logger.info(f"  📄 Results: {len(results)} documents")
    
    avg_time = sum(search_times) / len(search_times)
    logger.info(f"📈 {store_name} average search time: {avg_time:.3f}s")
    
    return avg_time

def main():
    """Compare ChromaDB vs Qdrant performance"""
    
    logger.info("🚀 Starting ChromaDB vs Qdrant performance comparison...")
    
    # Test queries
    test_queries = [
        "best study spots on ASU campus",
        "engineering programs at ASU",
        "campus dining options",
        "online classes and remote learning",
        "professor ratings and reviews"
    ]
    
    try:
        # Test ChromaDB
        logger.info("🔄 Testing ChromaDB...")
        chroma_store = ChromaVectorStore(Config.COLLECTION_NAME, Config.VECTOR_DB_DIR)
        chroma_time = test_search_performance(chroma_store, "ChromaDB", test_queries)
        
        # Test Qdrant
        logger.info("🔄 Testing Qdrant...")
        qdrant_store = QdrantStore(Config.COLLECTION_NAME)
        qdrant_time = test_search_performance(qdrant_store, "Qdrant", test_queries)
        
        # Compare results
        logger.info("📊 Performance Comparison Results:")
        logger.info(f"  ChromaDB: {chroma_time:.3f}s average")
        logger.info(f"  Qdrant: {qdrant_time:.3f}s average")
        
        if chroma_time and qdrant_time:
            speedup = chroma_time / qdrant_time
            if speedup > 1:
                logger.info(f"  🏆 Qdrant is {speedup:.1f}x faster than ChromaDB")
            else:
                logger.info(f"  🏆 ChromaDB is {1/speedup:.1f}x faster than Qdrant")
        
        # Memory usage comparison
        logger.info("💾 Storage Comparison:")
        chroma_stats = chroma_store.get_stats()
        qdrant_stats = qdrant_store.get_stats()
        
        logger.info(f"  ChromaDB: {chroma_stats}")
        logger.info(f"  Qdrant: {qdrant_stats}")
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        raise

if __name__ == "__main__":
    main() 