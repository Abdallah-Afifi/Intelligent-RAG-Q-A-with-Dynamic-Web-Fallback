"""
State definitions for the LangGraph workflow.
Defines the state structure that flows through the graph.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add


class GraphState(TypedDict):
    """
    State object that flows through the graph.
    
    Attributes:
        question: User's input question
        reformulated_query: Query reformulated for better search
        retrieved_documents: Documents retrieved from vector store
        retrieval_scores: Similarity scores for retrieved documents
        relevance_assessment: Assessment of document relevance
        is_knowledge_base_sufficient: Whether KB can answer the question
        needs_web_fallback: Whether to fallback to web after RAG answer
        rag_context: Formatted context from knowledge base
        rag_sources: Source metadata from knowledge base
        web_search_query: Query used for web search
        web_results: Results from web search
        web_sources: Web source metadata
        answer: Final generated answer
        source_type: Type of source used ('knowledge_base' or 'web')
        citations: Formatted citations
        error: Any error that occurred
        metadata: Additional metadata
    """
    # Input
    question: str
    
    # RAG Path
    reformulated_query: Optional[str]
    retrieved_documents: Annotated[List[Any], add]
    retrieval_scores: List[float]
    relevance_assessment: Dict[str, Any]
    is_knowledge_base_sufficient: bool
    needs_web_fallback: bool
    rag_context: Optional[str]
    rag_sources: List[Dict[str, Any]]
    
    # Web Path
    web_search_query: Optional[str]
    web_results: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    
    # Output
    answer: str
    source_type: str
    citations: str
    
    # Metadata
    error: Optional[str]
    metadata: Dict[str, Any]


def create_initial_state(question: str) -> GraphState:
    """
    Create initial state for the graph.
    
    Args:
        question: User's question
    
    Returns:
        Initial GraphState
    """
    return GraphState(
        question=question,
        reformulated_query=None,
        retrieved_documents=[],
        retrieval_scores=[],
        relevance_assessment={},
        is_knowledge_base_sufficient=False,
        needs_web_fallback=False,
        rag_context=None,
        rag_sources=[],
        web_search_query=None,
        web_results=[],
        web_sources=[],
        answer="",
        source_type="",
        citations="",
        error=None,
        metadata={},
    )
