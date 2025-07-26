#!/usr/bin/env python3
"""
Skip Build - Just Verify Qdrant Connection
This script just checks if Qdrant is accessible and has data
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config
from src.rag.qdrant_store import QdrantStore

def main():
    """Just verify Qdrant connection and data"""

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("🔍 Verifying Qdrant connection and data...")

    try:
        # Load configuration
        config = Config()
        logger.info("✅ Configuration loaded")

        # Test Qdrant connection
        qdrant_store = QdrantStore(config.COLLECTION_NAME)
        logger.info("✅ Connected to Qdrant cloud cluster")

        # Get stats
        stats = qdrant_store.get_stats()
        total_docs = stats.get('total_documents', 0)
        
        logger.info(f"📊 Qdrant collection stats: {stats}")
        
        if total_docs > 0:
            logger.info(f"✅ SUCCESS: Found {total_docs} documents in Qdrant!")
            logger.info("🚀 Production can now use this pre-populated database")
            return True
        else:
            logger.warning("⚠️ No documents found in Qdrant")
            logger.info("💡 You need to populate Qdrant from local first")
            return False

    except Exception as e:
        logger.error(f"❌ Error connecting to Qdrant: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 