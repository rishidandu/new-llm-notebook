import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models

from src.utils.data_processor import Document
from config.settings import Config

class QdrantStore:
    """Qdrant vector store implementation"""
    
    def __init__(self, collection_name: str, host: str = "localhost", port: int = 6333):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=host, port=port)
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI text-embedding-3-small dimension
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Created new Qdrant collection: {self.collection_name}")
            else:
                self.logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Error creating Qdrant collection: {e}")
            raise
    
    def _clean_metadata_for_qdrant(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata to be compatible with Qdrant"""
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = ""
            elif isinstance(value, (int, float, str, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents to Qdrant"""
        if not documents or not embeddings:
            self.logger.warning("No documents or embeddings provided")
            return
        
        # Prepare points for Qdrant
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            if embedding and isinstance(embedding, list) and len(embedding) > 0:
                # Clean metadata
                cleaned_metadata = self._clean_metadata_for_qdrant(doc.metadata)
                
                # Create point
                point = PointStruct(
                    id=doc.id,
                    vector=embedding,
                    payload={
                        'content': doc.content,
                        'source': doc.source,
                        **cleaned_metadata
                    }
                )
                points.append(point)
            else:
                self.logger.warning(f"Skipping document {doc.id} - invalid embedding")
        
        if not points:
            self.logger.warning("No valid points to add")
            return
        
        # Add points in batches
        batch_size = Config.BATCH_SIZE
        successful_adds = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                successful_adds += len(batch)
                self.logger.info(f"Added batch {i//batch_size + 1} to Qdrant ({len(batch)} documents)")
            except Exception as e:
                self.logger.error(f"Error adding batch {i//batch_size + 1} to Qdrant: {e}")
        
        self.logger.info(f"Successfully added {successful_adds} out of {len(documents)} documents to Qdrant")
    
    def search(self, query_embedding: List[float], top_k: int = 5, filter_conditions: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents in Qdrant"""
        try:
            # Build filter if conditions provided
            search_filter = None
            if filter_conditions:
                conditions = []
                for field, value in filter_conditions.items():
                    conditions.append(FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    ))
                search_filter = Filter(must=conditions)
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            formatted_results = []
            for result in search_result:
                formatted_results.append({
                    'content': result.payload.get('content', ''),
                    'metadata': {k: v for k, v in result.payload.items() if k != 'content'},
                    'score': result.score,
                    'rank': len(formatted_results) + 1
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching Qdrant: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'total_documents': collection_info.points_count,
                'collection_name': self.collection_name,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance.value
            }
        except Exception as e:
            self.logger.error(f"Error getting Qdrant collection stats: {e}")
            return {
                'total_documents': 0,
                'collection_name': self.collection_name
            }
    
    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Deleted Qdrant collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error deleting Qdrant collection: {e}")
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[],  # Empty list deletes all points
                )
            )
            self.logger.info(f"Cleared Qdrant collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error clearing Qdrant collection: {e}") 