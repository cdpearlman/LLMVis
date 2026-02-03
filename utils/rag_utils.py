"""
RAG (Retrieval-Augmented Generation) Utilities

Provides document loading, chunking, embedding, and retrieval for the AI chatbot.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from utils.openrouter_client import get_embedding, get_query_embedding, get_openrouter_client


# Configuration
CHUNK_SIZE = 500  # Target tokens per chunk (approximate)
CHUNK_OVERLAP = 50  # Overlap between chunks
SUPPORTED_EXTENSIONS = {'.txt', '.md'}
CACHE_FILE = "embeddings_cache.json"


class DocumentChunk:
    """Represents a chunk of a document with its embedding."""
    
    def __init__(
        self,
        content: str,
        source_file: str,
        chunk_index: int,
        embedding: Optional[List[float]] = None
    ):
        self.content = content
        self.source_file = source_file
        self.chunk_index = chunk_index
        self.embedding = embedding
        self.content_hash = hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "embedding": self.embedding,
            "content_hash": self.content_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentChunk':
        """Create from dictionary."""
        chunk = cls(
            content=data["content"],
            source_file=data["source_file"],
            chunk_index=data["chunk_index"],
            embedding=data.get("embedding")
        )
        chunk.content_hash = data.get("content_hash", chunk.content_hash)
        return chunk


class RAGService:
    """Service for managing RAG document retrieval."""
    
    def __init__(self, docs_path: str = "rag_docs"):
        """
        Initialize the RAG service.
        
        Args:
            docs_path: Path to the documents folder
        """
        self.docs_path = Path(docs_path)
        self.cache_path = self.docs_path / CACHE_FILE
        self.chunks: List[DocumentChunk] = []
        self._loaded = False
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        return len(text) // 4
    
    def _chunk_text(self, text: str, source_file: str) -> List[DocumentChunk]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text content to chunk
            source_file: Name of the source file
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size, save current and start new
            if self._estimate_tokens(current_chunk + para) > CHUNK_SIZE and current_chunk:
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    source_file=source_file,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
                
                # Keep overlap from the end of current chunk
                words = current_chunk.split()
                overlap_words = words[-CHUNK_OVERLAP:] if len(words) > CHUNK_OVERLAP else words
                current_chunk = " ".join(overlap_words) + "\n\n"
            
            current_chunk += para + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                source_file=source_file,
                chunk_index=chunk_index
            ))
        
        return chunks
    
    def load_documents(self) -> int:
        """
        Load and chunk all documents from the docs folder.
        
        Returns:
            Number of chunks loaded
        """
        if not self.docs_path.exists():
            print(f"RAG docs folder not found: {self.docs_path}")
            return 0
        
        # Try to load from cache first
        cached_chunks = self._load_cache()
        cached_hashes = {c.content_hash for c in cached_chunks}
        
        new_chunks = []
        
        # Load all document files
        for file_path in self.docs_path.iterdir():
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if file_path.name == CACHE_FILE or file_path.name.startswith('.'):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_chunks = self._chunk_text(content, file_path.name)
                
                for chunk in file_chunks:
                    if chunk.content_hash in cached_hashes:
                        # Use cached version with embedding
                        cached_chunk = next(
                            (c for c in cached_chunks if c.content_hash == chunk.content_hash),
                            None
                        )
                        if cached_chunk:
                            new_chunks.append(cached_chunk)
                    else:
                        new_chunks.append(chunk)
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        self.chunks = new_chunks
        self._loaded = True
        
        return len(self.chunks)
    
    def embed_documents(self) -> int:
        """
        Generate embeddings for all chunks that don't have them.
        
        Returns:
            Number of new embeddings generated
        """
        if not self._loaded:
            self.load_documents()
        
        client = get_openrouter_client()
        if not client.is_available:
            print("OpenRouter client not available, skipping embedding generation")
            return 0
        
        embedded_count = 0
        
        for chunk in self.chunks:
            if chunk.embedding is None:
                embedding = get_embedding(chunk.content)
                if embedding:
                    chunk.embedding = embedding
                    embedded_count += 1
        
        # Save to cache after embedding
        if embedded_count > 0:
            self._save_cache()
        
        return embedded_count
    
    def _load_cache(self) -> List[DocumentChunk]:
        """Load cached embeddings from file."""
        if not self.cache_path.exists():
            return []
        
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [DocumentChunk.from_dict(d) for d in data]
        except Exception as e:
            print(f"Error loading cache: {e}")
            return []
    
    def _save_cache(self):
        """Save embeddings to cache file."""
        try:
            data = [c.to_dict() for c in self.chunks if c.embedding is not None]
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: User's query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self._loaded:
            self.load_documents()
            self.embed_documents()
        
        # Get query embedding
        query_embedding = get_query_embedding(query)
        if query_embedding is None:
            return []
        
        query_vec = np.array(query_embedding)
        
        # Calculate similarities
        results = []
        for chunk in self.chunks:
            if chunk.embedding is None:
                continue
            
            chunk_vec = np.array(chunk.embedding)
            
            # Cosine similarity
            similarity = np.dot(query_vec, chunk_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-8
            )
            results.append((chunk, float(similarity)))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def build_context(self, query: str, top_k: int = 3) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            query: User's query
            top_k: Number of chunks to include
            
        Returns:
            Formatted context string for the prompt
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return ""
        
        context_parts = []
        for chunk, score in results:
            source = chunk.source_file
            context_parts.append(f"[From {source}]:\n{chunk.content}")
        
        return "\n\n---\n\n".join(context_parts)


# Singleton instance
_rag_instance: Optional[RAGService] = None


def get_rag_service(docs_path: str = "rag_docs") -> RAGService:
    """Get or create the singleton RAG service instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGService(docs_path)
    return _rag_instance


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
    """
    Convenience function to retrieve relevant chunks.
    
    Args:
        query: User's query
        top_k: Number of chunks to retrieve
        
    Returns:
        List of (chunk, score) tuples
    """
    service = get_rag_service()
    return service.retrieve(query, top_k)


def build_rag_context(query: str, top_k: int = 3) -> str:
    """
    Convenience function to build RAG context.
    
    Args:
        query: User's query
        top_k: Number of chunks to include
        
    Returns:
        Formatted context string
    """
    service = get_rag_service()
    return service.build_context(query, top_k)


def initialize_rag(docs_path: str = "rag_docs") -> int:
    """
    Initialize the RAG service by loading and embedding documents.
    
    Args:
        docs_path: Path to documents folder
        
    Returns:
        Number of chunks loaded
    """
    service = get_rag_service(docs_path)
    num_chunks = service.load_documents()
    service.embed_documents()
    return num_chunks
