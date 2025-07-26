#!/usr/bin/env python3
"""
Test Qdrant connection and basic operations
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config
from src.rag.qdrant_store import QdrantStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qdrant_connection():
    """Test Qdrant connection and basic operations"""
    
    logger.info("🔍 Testing Qdrant connection...")
    
    try:
        # Load config
        config = Config()
        logger.info(f"📋 Config loaded - QDRANT_HOST: {config.QDRANT_HOST}")
        
        # Test connection
        qdrant_store = QdrantStore("test_collection")
        logger.info("✅ Successfully connected to Qdrant!")
        
        # Test collection creation
        logger.info("🔄 Testing collection operations...")
        
        # Get stats
        stats = qdrant_store.get_stats()
        logger.info(f"📊 Collection stats: {stats}")
        
        # Test adding a simple document
        from src.utils.data_processor import Document
        
        test_doc = Document(
            id="test_1",
            content="This is a test document for ASU RAG system",
            metadata={"source": "test", "type": "sample"},
            source="test"
        )
        
        # Simple test embedding
        test_embedding = [0.1] * 1536
        
        logger.info("🔄 Adding test document...")
        qdrant_store.add_documents([test_doc], [test_embedding])
        
        # Test search
        logger.info("🔍 Testing search...")
        results = qdrant_store.search(test_embedding, top_k=1)
        logger.info(f"📄 Search results: {len(results)} documents found")
        
        # Get updated stats
        stats = qdrant_store.get_stats()
        logger.info(f"📊 Updated stats: {stats}")
        
        # Clean up
        logger.info("🧹 Cleaning up test collection...")
        qdrant_store.delete_collection()
        
        logger.info("✅ All Qdrant tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Qdrant test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_qdrant_connection()
    sys.exit(0 if success else 1) 