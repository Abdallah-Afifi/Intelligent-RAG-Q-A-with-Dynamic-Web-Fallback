"""RAG retrieval pipeline for document retrieval and relevance assessment."""

from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
from src.vector_store.chroma_store import ChromaVectorStore
from src.utils.logger import app_logger
from src.utils.helpers import calculate_relevance_score, Timer
from config.settings import settings


class RAGRetriever:
    """Handles retrieval operations for RAG pipeline."""
    
    def __init__(self, vector_store: ChromaVectorStore):
        """Initialize RAG retriever."""
        self.vector_store = vector_store
        self.relevance_threshold = settings.RELEVANCE_THRESHOLD
        self.min_confidence = settings.MIN_CONFIDENCE_SCORE
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Tuple[List[Document], List[float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
        
        Returns:
            Tuple of (documents, scores)
        """
        top_k = top_k or settings.TOP_K_RETRIEVAL
        
        app_logger.info(f"Retrieving documents for query: {query[:100]}...")
        
        with Timer("Document retrieval"):
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        if not results:
            app_logger.warning("No documents retrieved")
            return [], []
        
        # Separate documents and scores
        documents = [doc for doc, _ in results]
        scores = [score for _, score in results]
        
        # ChromaDB returns L2 distance, convert to similarity (lower distance => higher similarity)
        # Use bounded conversion: similarity = 1 / (1 + distance)
        normalized_scores = [1.0 / (1.0 + d) for d in scores]
        
        app_logger.info(f"Retrieved {len(documents)} documents")
        app_logger.debug(f"Similarity scores: {normalized_scores}")
        
        return documents, normalized_scores
    
    def assess_relevance(
        self,
        query: str,
        documents: List[Document],
        scores: List[float]
    ) -> Dict[str, Any]:
        """
        Assess whether retrieved documents are relevant to the query.
        
        Args:
            query: User query
            documents: Retrieved documents
            scores: Similarity scores
        
        Returns:
            Dictionary with relevance assessment
        """
        if not documents or not scores:
            return {
                'is_relevant': False,
                'confidence': 0.0,
                'reason': 'No documents retrieved',
                'top_score': 0.0,
                'avg_score': 0.0,
            }
        
        # The provided 'scores' are similarity scores (0-1). Reuse directly.
        similarity_scores = scores
        
        # Calculate metrics
        top_score = max(similarity_scores)
        avg_score = sum(similarity_scores) / len(similarity_scores)
        overall_score = calculate_relevance_score(similarity_scores)
        
        # Determine relevance
        is_relevant = (
            top_score >= self.relevance_threshold and
            overall_score >= self.min_confidence
        )
        
        # Generate reason
        if is_relevant:
            reason = f"High relevance detected (top similarity: {top_score:.2f})"
        else:
            if top_score < self.relevance_threshold:
                reason = f"Top similarity {top_score:.2f} below threshold {self.relevance_threshold}"
            else:
                reason = f"Overall confidence {overall_score:.2f} below minimum {self.min_confidence}"
        
        result = {
            'is_relevant': is_relevant,
            'confidence': overall_score,
            'reason': reason,
            'top_score': top_score,
            'avg_score': avg_score,
            'num_documents': len(documents),
        }
        
        app_logger.info(
            f"Relevance assessment: {result['is_relevant']} "
            f"(confidence: {result['confidence']:.2f})"
        )
        
        return result
    
    def retrieve_and_assess(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Retrieve documents and assess their relevance.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
        
        Returns:
            Tuple of (documents, assessment)
        """
        documents, scores = self.retrieve(query, top_k)
        assessment = self.assess_relevance(query, documents, scores)
        
        return documents, assessment
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of documents
        
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            page = doc.metadata.get('page', 'Unknown')
            content = doc.page_content.strip()
            context_parts.append(
                f"[Document {i} - Page {page}]\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def get_source_metadata(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract source metadata from documents.
        
        Args:
            documents: List of documents
        
        Returns:
            List of source metadata dictionaries
        """
        sources = []
        seen_pages = set()
        
        for doc in documents:
            page = doc.metadata.get('page')
            if page and page not in seen_pages:
                sources.append({
                    'page': page,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'preview': doc.page_content[:200] + "..."
                })
                seen_pages.add(page)
        
        return sources
