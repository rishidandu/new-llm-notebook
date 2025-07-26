#!/usr/bin/env python3
"""
Build RAG System with Optimized Data Processing
Uses conversation context and quality scoring for better retrieval
"""

import sys
import os
from typing import List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.rag_optimized_processor import RAGOptimizedProcessor
from src.utils.data_processor import Document
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from config.settings import Config
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Build RAG system with optimized processing"""
    config = Config()
    
    print("🚀 Building RAG system with optimized processing...")
    start_time = time.time()
    
    # Initialize components
    processor = RAGOptimizedProcessor(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore(config.COLLECTION_NAME, config.VECTOR_DB_DIR)
    
    # Get latest data files
    def get_latest_files(directory: str) -> List[str]:
        """Get the most recent data files from a directory"""
        if not os.path.exists(directory):
            return []
        
        files = []
        for filename in os.listdir(directory):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(directory, filename)
                files.append(file_path)
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return files
    
    reddit_files = get_latest_files(config.REDDIT_RAW_DIR)
    asu_files = get_latest_files(config.ASU_RAW_DIR)
    
    print(f"📁 Found {len(reddit_files)} Reddit files and {len(asu_files)} ASU files")
    
    # If no data files found, create sample data for testing
    if not reddit_files and not asu_files:
        print("⚠️ No data files found. Creating sample data for testing...")
        create_sample_data()
        reddit_files = get_latest_files(config.REDDIT_RAW_DIR)
        asu_files = get_latest_files(config.ASU_RAW_DIR)
        print(f"📁 Created sample data: {len(reddit_files)} Reddit files and {len(asu_files)} ASU files")
    
    total_documents = 0
    
    # Process Reddit data with conversation context
    for file_path in reddit_files:
        print(f"🔄 Processing Reddit file: {os.path.basename(file_path)}")
        
        # Option 1: Process with conversation context
        documents = list(processor.process_reddit_data_rag_optimized(file_path))
        
        # Option 2: Process as conversation threads (alternative)
        # documents = list(processor.process_conversation_threads(file_path))
        
        print(f"   📄 Generated {len(documents)} RAG-optimized documents")
        
        # Add to vector store with quality scoring
        if documents:
            # Generate embeddings for all documents
            embeddings = []
            enhanced_documents = []
            
            for doc in documents:
                # Generate embedding
                embedding = embedding_generator.get_embedding(doc.content)
                embeddings.append(embedding)
                
                # Add with quality score in metadata
                enhanced_metadata = {
                    **doc.metadata,
                    'quality_score': doc.quality_score,
                    'conversation_context': doc.conversation_context
                }
                
                # Create enhanced document
                enhanced_doc = Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=enhanced_metadata,
                    source=doc.source
                )
                enhanced_documents.append(enhanced_doc)
            
            # Add all documents at once
            vector_store.add_documents(enhanced_documents, embeddings)
        
        total_documents += len(documents)
    
    # Process ASU data (standard processing)
    for file_path in asu_files:
        print(f"🔄 Processing ASU file: {os.path.basename(file_path)}")
        
        # Use standard processing for ASU data
        from src.utils.data_processor import DataProcessor
        standard_processor = DataProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        documents = list(standard_processor.process_asu_data(file_path))
        print(f"   📄 Generated {len(documents)} documents")
        
        if documents:
            # Generate embeddings for all documents
            embeddings = []
            for doc in documents:
                embedding = embedding_generator.get_embedding(doc.content)
                embeddings.append(embedding)
            
            # Add all documents at once
            vector_store.add_documents(documents, embeddings)
        
        total_documents += len(documents)
    
    # Vector store is automatically persisted
    
    elapsed_time = time.time() - start_time
    print(f"✅ RAG system built successfully!")
    print(f"📊 Total documents processed: {total_documents}")
    print(f"⏱️ Total time: {elapsed_time:.2f} seconds")
    print(f"🚀 Documents per second: {total_documents/elapsed_time:.1f}")

def create_sample_data():
    """Create sample data for testing when no real data is available"""
    import json
    import os
    
    print("📝 Creating sample Reddit data...")
    
    # Sample Reddit data with ASU-related content
    sample_reddit_data = [
        {
            "id": "sample_1",
            "title": "Best study spots on ASU campus",
            "text": "The Hayden Library is one of the best study spots on ASU campus. It has multiple floors with quiet study areas, group study rooms, and 24/7 access during finals week. The Noble Library is also great for engineering students. The Memorial Union has good spots for group study sessions. The Tempe campus has many great study locations including the Student Services Building and the Design School.",
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
            "text": "Mill Avenue is the main entertainment district near ASU with restaurants, bars, and shops. The Tempe Town Lake is great for outdoor activities like kayaking and paddleboarding. The ASU Art Museum is free for students and has great exhibits. The Sun Devil Stadium area is perfect for game days and events. The Memorial Union has food courts and study spaces. The campus is beautiful for walking and exploring.",
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
        },
        {
            "id": "sample_3",
            "title": "Campus dining options and meal plans",
            "text": "ASU has several dining halls including the Memorial Union, Barrett Dining Center, and Hassayampa Dining. There are also food courts, coffee shops, and restaurants throughout campus. Meal plans are available for students living on campus. The Memorial Union has Chick-fil-A, Panda Express, and other popular chains. The campus also has many local restaurants and food trucks. The meal plans are convenient for students who don't want to cook.",
            "url": "https://reddit.com/r/ASU/comments/sample3",
            "source": "reddit",
            "ingested_at": "2025-07-26T00:00:00Z",
            "metadata": {
                "subreddit": "ASUStudents",
                "score": 28,
                "num_comments": 15,
                "author": "food_lover",
                "post_type": "submission"
            }
        },
        {
            "id": "sample_4",
            "title": "ASU online classes and remote learning",
            "text": "ASU Online offers many degree programs that can be completed entirely online. The online platform is user-friendly and includes video lectures, discussion boards, and virtual office hours. Many students appreciate the flexibility of online classes. The professors are responsive and provide good feedback. The online library resources are extensive. ASU has been a leader in online education for years.",
            "url": "https://reddit.com/r/ASUOnline/comments/sample4",
            "source": "reddit",
            "ingested_at": "2025-07-26T00:00:00Z",
            "metadata": {
                "subreddit": "ASUOnline",
                "score": 67,
                "num_comments": 23,
                "author": "online_learner",
                "post_type": "submission"
            }
        },
        {
            "id": "sample_5",
            "title": "ASU engineering programs and research",
            "text": "ASU's Ira A. Fulton Schools of Engineering is one of the largest engineering schools in the country. They offer programs in mechanical, electrical, computer science, civil, and many other engineering disciplines. The research opportunities are excellent, especially in areas like robotics, renewable energy, and artificial intelligence. The engineering buildings have state-of-the-art labs and equipment. Many engineering students get internships and job offers from top companies.",
            "url": "https://reddit.com/r/ASUEngineering/comments/sample5",
            "source": "reddit",
            "ingested_at": "2025-07-26T00:00:00Z",
            "metadata": {
                "subreddit": "ASUEngineering",
                "score": 89,
                "num_comments": 31,
                "author": "engineering_student",
                "post_type": "submission"
            }
        }
    ]
    
    # Create directories if they don't exist
    os.makedirs("data/raw/reddit", exist_ok=True)
    os.makedirs("data/raw/asu_web", exist_ok=True)
    
    # Write sample Reddit data
    with open("data/raw/reddit/sample_data.jsonl", "w") as f:
        for item in sample_reddit_data:
            f.write(json.dumps(item) + "\n")
    
    # Create sample ASU web data
    sample_asu_data = [
        {
            "id": "asu_web_1",
            "title": "ASU Campus Life and Student Services",
            "text": "Arizona State University offers comprehensive student services including academic advising, career counseling, health services, and mental health support. The campus has numerous student organizations and clubs for every interest. The Sun Devil Fitness Complex provides excellent workout facilities. The university also offers housing assistance, financial aid counseling, and international student services. ASU is committed to student success and provides many resources to help students thrive.",
            "url": "https://www.asu.edu/student-life",
            "source": "asu_web",
            "ingested_at": "2025-07-26T00:00:00Z"
        },
        {
            "id": "asu_web_2",
            "title": "ASU Academic Programs and Degrees",
            "text": "ASU offers over 350 undergraduate majors and more than 400 graduate degree programs across its four campuses. The university is known for innovation and interdisciplinary research. Popular programs include business, engineering, computer science, psychology, and biological sciences. ASU also offers many online degree programs through ASU Online. The university has partnerships with industry leaders and provides excellent internship and job placement opportunities.",
            "url": "https://www.asu.edu/academics",
            "source": "asu_web", 
            "ingested_at": "2025-07-26T00:00:00Z"
        }
    ]
    
    # Write sample ASU web data
    with open("data/raw/asu_web/sample_data.jsonl", "w") as f:
        for item in sample_asu_data:
            f.write(json.dumps(item) + "\n")
    
    print("✅ Created sample data files for testing")

if __name__ == "__main__":
    main() 