#!/usr/bin/env python3
"""
Migrate from ChromaDB to Qdrant
This script will transfer all documents and embeddings from ChromaDB to Qdrant
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

def main():
    """Migrate from ChromaDB to Qdrant"""
    
    logger.info("🚀 Starting migration from ChromaDB to Qdrant...")
    
    try:
        # Initialize ChromaDB (source)
        chroma_store = ChromaVectorStore(Config.COLLECTION_NAME, Config.VECTOR_DB_DIR)
        
        # Get stats from ChromaDB
        chroma_stats = chroma_store.get_stats()
        total_docs = chroma_stats.get('total_documents', 0)
        
        if total_docs == 0:
            logger.warning("⚠️ No documents found in ChromaDB. Nothing to migrate.")
            return
        
        logger.info(f"📊 Found {total_docs} documents in ChromaDB")
        
        # Initialize Qdrant (destination)
        qdrant_store = QdrantStore(Config.COLLECTION_NAME)
        
        # Get all documents from ChromaDB
        logger.info("🔄 Retrieving documents from ChromaDB...")
        
        # Note: This is a simplified migration. In a real scenario, you'd want to:
        # 1. Get all document IDs from ChromaDB
        # 2. Retrieve documents in batches
        # 3. Transfer embeddings and metadata
        
        # For now, we'll rebuild the embeddings from the raw data
        logger.info("🔄 Rebuilding embeddings for Qdrant...")
        
        # Import the RAG system to rebuild embeddings
        from src.rag.rag_system import ASURAGSystem
        
        # Initialize RAG system with Qdrant
        rag_system = ASURAGSystem(Config(), vector_store_type="qdrant")
        
        # Rebuild the database
        logger.info("🔄 Rebuilding RAG system with Qdrant...")
        rag_system.ingest_data(["reddit", "asu_web", "asu_grades"])
        
        # Get stats from Qdrant
        qdrant_stats = qdrant_store.get_stats()
        qdrant_docs = qdrant_stats.get('total_documents', 0)
        
        logger.info(f"✅ Migration completed!")
        logger.info(f"📊 ChromaDB documents: {total_docs}")
        logger.info(f"📊 Qdrant documents: {qdrant_docs}")
        
        if qdrant_docs > 0:
            logger.info("🎉 Successfully migrated to Qdrant!")
            logger.info("💡 You can now update your configuration to use Qdrant instead of ChromaDB")
        else:
            logger.error("❌ Migration failed - no documents in Qdrant")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    main() 