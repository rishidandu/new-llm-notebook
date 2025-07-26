#!/usr/bin/env python3
"""
Build RAG system for Render deployment
This script will populate the vector database with ASU data
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config
from src.rag.rag_system import ASURAGSystem

def main():
    """Build RAG system for production"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Building RAG system for Render deployment...")
    
    try:
        # Load configuration
        config = Config()
        logger.info("✅ Configuration loaded")
        
        # Initialize RAG system
        rag_system = ASURAGSystem(config)
        logger.info("✅ RAG system initialized")
        
        # Check what data sources are available
        data_sources = []
        
        # Check for Reddit data
        reddit_files = list(Path("data/raw/reddit").glob("*.jsonl"))
        if reddit_files:
            data_sources.append("reddit")
            logger.info(f"📊 Found {len(reddit_files)} Reddit data files")
        
        # Check for ASU grades data
        grades_dir = Path("data/raw/raw_asu_grades")
        if grades_dir.exists() and any(grades_dir.glob("*.csv")):
            data_sources.append("asu_grades")
            logger.info("📊 Found ASU grades data")
        
        # Check for ASU web data
        asu_web_files = list(Path("data/raw/asu_web").glob("*.jsonl"))
        if asu_web_files:
            data_sources.append("asu_web")
            logger.info(f"📊 Found {len(asu_web_files)} ASU web data files")
        
        if not data_sources:
            logger.warning("⚠️ No data sources found. Creating sample data...")
            # Create some sample data for testing
            create_sample_data()
            data_sources = ["sample"]
        
        # Ingest data
        logger.info(f"🔄 Ingesting data from sources: {data_sources}")
        rag_system.ingest_data(data_sources)
        
        # Verify the build
        stats = rag_system.get_stats()
        total_docs = stats.get('vector_store', {}).get('total_documents', 0)
        
        logger.info(f"✅ RAG system built successfully!")
        logger.info(f"📊 Total documents: {total_docs}")
        logger.info(f"🔧 Configuration: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error building RAG system: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    import json
    
    # Create sample Reddit data
    sample_reddit_data = [
        {
            "title": "Best study spots on ASU campus",
            "content": "The Hayden Library is one of the best study spots on ASU campus. It has multiple floors with quiet study areas, group study rooms, and 24/7 access during finals week. The Noble Library is also great for engineering students. The Memorial Union has good spots for group study sessions.",
            "url": "https://reddit.com/r/ASU/comments/sample1",
            "source": "reddit"
        },
        {
            "title": "Cool places to hang out around ASU",
            "content": "Mill Avenue is the main entertainment district near ASU with restaurants, bars, and shops. The Tempe Town Lake is great for outdoor activities. The ASU Art Museum is free for students. The Sun Devil Stadium area is perfect for game days and events.",
            "url": "https://reddit.com/r/ASU/comments/sample2", 
            "source": "reddit"
        },
        {
            "title": "Campus dining options",
            "content": "ASU has several dining halls including the Memorial Union, Barrett Dining Center, and Hassayampa Dining. There are also food courts, coffee shops, and restaurants throughout campus. Meal plans are available for students living on campus.",
            "url": "https://reddit.com/r/ASU/comments/sample3",
            "source": "reddit"
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