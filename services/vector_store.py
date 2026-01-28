"""
Vector Store Service
ChromaDB integration with embeddings for document storage and retrieval.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_PATH = Path(os.path.abspath("./chroma_db"))
COLLECTION_NAME = "tax_documents"

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Manages document storage and retrieval using ChromaDB with embeddings.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to reuse the same ChromaDB client."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Ensure persistence directory exists
        CHROMA_DB_PATH.mkdir(exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Set up embedding function
        openai_api_key = os.getenv("OPENAI_API_KEY")
        use_openai = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
        
        if openai_api_key and use_openai:
            try:
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name="text-embedding-3-small"
                )
                logger.info("Using OpenAI embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI embeddings: {e}, using default")
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        else:
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            logger.info("Using default local embeddings (sentence-transformers)")
        
        # Cache for loaded collections
        self.collections = {}
        
        # Initialize default collection
        self.get_collection(COLLECTION_NAME)
        
        self._initialized = True
    
    def get_collection(self, name: str):
        """Get or create a collection by name."""
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function,
                metadata={"description": f"Collection {name} for RAG"}
            )
        return self.collections[name]
    
    def document_exists(self, file_hash: str, collection_name: str = COLLECTION_NAME) -> bool:
        """Check if a document exists in the specified collection."""
        try:
            collection = self.get_collection(collection_name)
            results = collection.get(
                ids=[file_hash],
                include=["metadatas"]
            )
            return len(results["ids"]) > 0
        except Exception as e:
            logger.warning(f"Error checking document existence: {e}")
            return False
    
    def store_document(
        self,
        file_hash: str,
        original_filename: str,
        document_type: str,
        markdown_content: str,
        extracted_fields: Dict[str, Any],
        raw_chunks: List[Dict[str, Any]],
        overall_confidence: float,
        needs_verification: bool,
        saved_path: str = "",
        collection_name: str = COLLECTION_NAME
    ) -> bool:
        """Store a parsed document in the specified collection."""
        try:
            collection = self.get_collection(collection_name)
            
            # Prepare metadata
            metadata = {
                "original_filename": original_filename,
                "document_type": document_type,
                "overall_confidence": overall_confidence,
                "needs_verification": needs_verification,
                "extracted_fields_json": json.dumps(extracted_fields),
                "raw_chunks_json": json.dumps(raw_chunks),
                "saved_path": saved_path,  # Actual file path in uploads/
            }
            
            # Store the document
            collection.upsert(
                ids=[file_hash],
                documents=[markdown_content],
                metadatas=[metadata]
            )
            
            logger.info(f"Stored document {original_filename} in {collection_name} with hash {file_hash[:12]}...")
            return True
            
        except Exception as e:
            logger.warning(f"Error storing document in vector store: {e}")
            return False
    
    def get_document(self, file_hash: str, collection_name: str = COLLECTION_NAME) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its hash from the specified collection."""
        try:
            collection = self.get_collection(collection_name)
            results = collection.get(
                ids=[file_hash],
                include=["documents", "metadatas"]
            )
            
            if not results["ids"]:
                return None
            
            metadata = results["metadatas"][0]
            
            # Reconstruct the document data
            return {
                "file_hash": file_hash,
                "original_filename": metadata.get("original_filename", ""),
                "document_type": metadata.get("document_type", ""),
                "markdown_content": results["documents"][0],
                "overall_confidence": metadata.get("overall_confidence", 0.0),
                "needs_verification": metadata.get("needs_verification", True),
                "extracted_fields": json.loads(metadata.get("extracted_fields_json", "{}")),
                "raw_chunks": json.loads(metadata.get("raw_chunks_json", "[]")),
                "saved_path": metadata.get("saved_path", ""),
            }
            
        except Exception as e:
            print(f"Error retrieving document: {e}")
            return None
    
    def search_documents(self, query: str, n_results: int = 5, collection_name: str = COLLECTION_NAME) -> List[Dict[str, Any]]:
        """Search for documents in the specified collection."""
        try:
            collection = self.get_collection(collection_name)
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            documents = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    documents.append({
                        "file_hash": doc_id,
                        "original_filename": metadata.get("original_filename", ""),
                        "document_type": metadata.get("document_type", ""),
                        "markdown_content": results["documents"][0][i],
                        "relevance_distance": results["distances"][0][i] if results["distances"] else None,
                        "extracted_fields": json.loads(metadata.get("extracted_fields_json", "{}")),
                        "overall_confidence": metadata.get("overall_confidence", 0.0), # Added for convenience
                    })
            
            return documents
            
        except Exception as e:
            print(f"Error searching documents in {collection_name}: {e}")
            return []
    
    def get_all_documents(self, collection_name: str = COLLECTION_NAME) -> List[Dict[str, Any]]:
        """Get all stored documents from the specified collection."""
        try:
            collection = self.get_collection(collection_name)
            results = collection.get(
                include=["documents", "metadatas"]
            )
            
            documents = []
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                documents.append({
                    "file_hash": doc_id,
                    "original_filename": metadata.get("original_filename", ""),
                    "document_type": metadata.get("document_type", ""),
                    "overall_confidence": metadata.get("overall_confidence", 0.0),
                    "needs_verification": metadata.get("needs_verification", True),
                    "extracted_fields": json.loads(metadata.get("extracted_fields_json", "{}")),
                    "saved_path": metadata.get("saved_path", ""),
                })
            
            return documents
            
        except Exception as e:
            print(f"Error getting all documents from {collection_name}: {e}")
            return []
    
    def delete_document(self, file_hash: str, collection_name: str = COLLECTION_NAME) -> bool:
        """Delete a document from the specified collection."""
        try:
            collection = self.get_collection(collection_name)
            collection.delete(ids=[file_hash])
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def get_document_count(self, collection_name: str = COLLECTION_NAME) -> int:
        """Get the total number of documents in the specified collection."""
        try:
            collection = self.get_collection(collection_name)
            return collection.count()
        except Exception:
            return 0


# Singleton instance
_vector_store: Optional[VectorStoreService] = None


def get_vector_store() -> VectorStoreService:
    """Get the singleton VectorStoreService instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store


# =============================================================================
# Modular Query Functions (Designed for LangChain Tool Conversion)
# =============================================================================

def query_tax_documents(
    query: str,
    top_k: int = 5,
    include_fields: bool = True,
    min_confidence: float = 0.0
) -> dict:
    """
    Query the tax document vector store for relevant documents.
    
    This function is designed to be easily convertible to a LangChain tool.
    It takes a natural language query and returns the most relevant tax documents
    from the ChromaDB vector store.
    
    Args:
        query: Natural language query to search for (e.g., "tuition fees paid in 2025")
        top_k: Number of top results to return (default: 5)
        include_fields: Whether to include extracted field details (default: True)
        min_confidence: Minimum confidence threshold to filter results (default: 0.0)
    
    Returns:
        dict with keys:
            - success: bool indicating if the query was successful
            - query: the original query string
            - num_results: number of results found
            - results: list of document matches, each containing:
                - rank: 1-indexed ranking
                - document_type: type of tax document
                - filename: original filename
                - similarity_score: 0-1 score (higher is more similar)
                - confidence: extraction confidence score
                - content_preview: first 500 chars of markdown content
                - extracted_fields: dict of field name -> value (if include_fields=True)
            - error: error message if success is False
    
    Example:
        >>> result = query_tax_documents("employment income from T4")
        >>> for doc in result["results"]:
        ...     print(f"{doc['rank']}. {doc['document_type']} - {doc['similarity_score']:.2%}")
    """
    try:
        vector_store = get_vector_store()
        
        # Query ChromaDB
        raw_results = vector_store.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Handle empty results
        if not raw_results["ids"] or not raw_results["ids"][0]:
            return {
                "success": True,
                "query": query,
                "num_results": 0,
                "results": [],
                "message": "No documents found matching your query."
            }
        
        results = []
        for i, doc_id in enumerate(raw_results["ids"][0]):
            metadata = raw_results["metadatas"][0][i]
            distance = raw_results["distances"][0][i] if raw_results["distances"] else 0
            
            # Convert distance to similarity score (ChromaDB uses L2 distance by default)
            # Lower distance = higher similarity, so we use 1/(1+distance)
            similarity_score = 1 / (1 + distance)
            
            # Get confidence and apply filter
            confidence = metadata.get("overall_confidence", 0.0)
            if confidence < min_confidence:
                continue
            
            # Build result entry
            result_entry = {
                "rank": len(results) + 1,
                "document_type": metadata.get("document_type", "Unknown"),
                "filename": metadata.get("original_filename", "Unknown"),
                "file_hash": doc_id,
                "similarity_score": round(similarity_score, 4),
                "distance": round(distance, 4),
                "confidence": round(confidence, 4),
                "content_preview": raw_results["documents"][0][i][:500] + "..." if len(raw_results["documents"][0][i]) > 500 else raw_results["documents"][0][i],
            }
            
            # Include extracted fields if requested
            if include_fields:
                import json
                fields_json = metadata.get("extracted_fields_json", "{}")
                try:
                    extracted_fields = json.loads(fields_json)
                    # Simplify to just field_name: value
                    result_entry["extracted_fields"] = {
                        k: v.get("value", v) if isinstance(v, dict) else v
                        for k, v in extracted_fields.items()
                    }
                except json.JSONDecodeError:
                    result_entry["extracted_fields"] = {}
            
            results.append(result_entry)
        
        return {
            "success": True,
            "query": query,
            "num_results": len(results),
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "num_results": 0,
            "results": [],
            "error": str(e)
        }


def get_document_summary() -> dict:
    """
    Get a summary of all documents in the vector store.
    
    Useful for understanding what data is available before querying.
    
    Returns:
        dict with keys:
            - total_documents: count of stored documents
            - documents: list of document summaries (type, filename, confidence)
    """
    try:
        vector_store = get_vector_store()
        all_docs = vector_store.get_all_documents()
        
        return {
            "success": True,
            "total_documents": len(all_docs),
            "documents": [
                {
                    "document_type": doc.get("document_type", "Unknown"),
                    "filename": doc.get("original_filename", "Unknown"),
                    "confidence": doc.get("overall_confidence", 0.0),
                    "needs_verification": doc.get("needs_verification", True)
                }
                for doc in all_docs
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "total_documents": 0,
            "documents": [],
            "error": str(e)
        }

