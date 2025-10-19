"""
LangGraph nodes - individual processing steps in the workflow.
Each node performs a specific task and updates the state.
"""

from typing import Dict, Any
import re
from src.graph.state import GraphState
from src.rag.retriever import RAGRetriever
from src.rag.answer_generator import AnswerGenerator
from src.web_search.web_searcher import WebSearcher
from src.utils.logger import app_logger
from config.settings import settings


def _answer_indicates_insufficient_info(answer: str) -> bool:
    """
    Check if the answer indicates insufficient information in the context.
    
    Args:
        answer: The generated answer text
        
    Returns:
        True if answer indicates lack of information
    """
    # Patterns that indicate the context doesn't have the answer
    insufficient_patterns = [
        r"(?i)don'?t have.*information",
        r"(?i)does not contain.*information",
        r"(?i)not mentioned in.*context",
        r"(?i)context does not.*enough",
        r"(?i)unable to provide",
        r"(?i)cannot answer",
        r"(?i)no information about",
        r"(?i)not included in.*context",
        r"(?i)unfortunately.*context",
    ]
    
    for pattern in insufficient_patterns:
        if re.search(pattern, answer):
            return True
    
    return False


class WorkflowNodes:
    """Collection of nodes for the LangGraph workflow."""
    
    def __init__(
        self,
        retriever: RAGRetriever,
        answer_generator: AnswerGenerator,
        web_searcher: WebSearcher
    ):
        """
        Initialize workflow nodes.
        
        Args:
            retriever: RAG retriever instance
            answer_generator: Answer generator instance
            web_searcher: Web searcher instance
        """
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.web_searcher = web_searcher
    
    def retrieve_from_kb(self, state: GraphState) -> GraphState:
        """
        Node: Retrieve documents from knowledge base.
        
        Args:
            state: Current graph state
        
        Returns:
            Updated state
        """
        app_logger.info("NODE: Retrieving from knowledge base")
        
        try:
            question = state['question']
            
            # Retrieve documents and assess relevance
            documents, assessment = self.retriever.retrieve_and_assess(question)
            
            # Update state
            state['retrieved_documents'] = documents
            state['retrieval_scores'] = [assessment['top_score']]
            state['relevance_assessment'] = assessment
            state['is_knowledge_base_sufficient'] = assessment['is_relevant']
            
            if documents:
                state['rag_context'] = self.retriever.format_context(documents)
                state['rag_sources'] = self.retriever.get_source_metadata(documents)
            
            app_logger.info(
                f"Retrieved {len(documents)} documents. "
                f"Sufficient: {assessment['is_relevant']}"
            )
        
        except Exception as e:
            app_logger.error(f"Error in retrieve_from_kb node: {str(e)}")
            state['error'] = str(e)
            state['is_knowledge_base_sufficient'] = False
        
        return state
    
    def generate_rag_answer(self, state: GraphState) -> GraphState:
        """
        Node: Generate answer from knowledge base context.
        
        Args:
            state: Current graph state
        
        Returns:
            Updated state
        """
        app_logger.info("NODE: Generating RAG answer")
        
        try:
            result = self.answer_generator.generate_rag_answer(
                question=state['question'],
                context=state['rag_context'],
                sources=state['rag_sources']
            )
            
            state['answer'] = result['answer']
            state['source_type'] = result['source_type']
            state['citations'] = result['citations']
            
            # Check if the answer indicates insufficient information
            if _answer_indicates_insufficient_info(result['answer']):
                app_logger.info("Answer indicates insufficient information - marking for web fallback")
                state['is_knowledge_base_sufficient'] = False
                state['needs_web_fallback'] = True
            
            app_logger.info("RAG answer generated successfully")
        
        except Exception as e:
            app_logger.error(f"Error in generate_rag_answer node: {str(e)}")
            state['error'] = str(e)
            state['answer'] = "Error generating answer from knowledge base."
        
        return state
    
    def notify_user_fallback(self, state: GraphState) -> GraphState:
        """
        Node: Notify user that we're falling back to web search.
        
        Args:
            state: Current graph state
        
        Returns:
            Updated state
        """
        app_logger.info("NODE: Notifying user of web fallback")
        
        # This is where we'd send a notification to the user
        # For now, we just log it and add metadata
        state['metadata']['fallback_notification'] = (
            "⚠️ The information was not found in the knowledge base. "
            "Searching the web for an answer..."
        )
        
        app_logger.info("User notified of web fallback")
        return state
    
    def search_web(self, state: GraphState) -> GraphState:
        """
        Node: Perform web search.
        
        Args:
            state: Current graph state
        
        Returns:
            Updated state
        """
        app_logger.info("NODE: Searching web")
        
        try:
            # Reformulate query for better web search
            reformulated = self.answer_generator.reformulate_query_for_web(
                state['question']
            )
            state['web_search_query'] = reformulated
            
            # Perform web search
            results = self.web_searcher.search_and_extract(
                query=reformulated,
                max_results=settings.MAX_SEARCH_RESULTS,
                extract_content=True
            )
            
            state['web_results'] = results
            app_logger.info(f"Found {len(results)} web results")
        
        except Exception as e:
            app_logger.error(f"Error in search_web node: {str(e)}")
            state['error'] = str(e)
            state['web_results'] = []
        
        return state
    
    def generate_web_answer(self, state: GraphState) -> GraphState:
        """
        Node: Generate answer from web search results.
        
        Args:
            state: Current graph state
        
        Returns:
            Updated state
        """
        app_logger.info("NODE: Generating web answer")
        
        try:
            if not state['web_results']:
                state['answer'] = (
                    "I apologize, but I couldn't find sufficient information "
                    "in either the knowledge base or the web to answer your question."
                )
                state['source_type'] = "none"
                return state
            
            result = self.answer_generator.generate_web_answer(
                question=state['question'],
                web_results=state['web_results']
            )
            
            state['answer'] = result['answer']
            state['source_type'] = result['source_type']
            state['citations'] = result['citations']
            state['web_sources'] = result['sources']
            
            app_logger.info("Web answer generated successfully")
        
        except Exception as e:
            app_logger.error(f"Error in generate_web_answer node: {str(e)}")
            state['error'] = str(e)
            state['answer'] = "Error generating answer from web sources."
        
        return state
    
    def format_final_output(self, state: GraphState) -> GraphState:
        """
        Node: Format final output with all metadata.
        
        Args:
            state: Current graph state
        
        Returns:
            Updated state
        """
        app_logger.info("NODE: Formatting final output")
        
        # Add metadata about the response
        state['metadata']['confidence'] = state['relevance_assessment'].get('confidence', 0.0)
        state['metadata']['num_sources'] = len(state.get('rag_sources', []) or state.get('web_sources', []))
        
        app_logger.info(f"Final answer prepared (source: {state['source_type']})")
        return state
