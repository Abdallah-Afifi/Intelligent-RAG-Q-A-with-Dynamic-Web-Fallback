"""
Answer generator for both RAG and web-based responses.
"""

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_core.language_models.chat_models import BaseChatModel
from src.prompts.templates import (
    get_rag_answer_prompt,
    get_web_answer_prompt,
    get_query_reformulation_prompt
)
from src.utils.logger import app_logger
from src.utils.helpers import format_citations, Timer


class AnswerGenerator:
    """Generates answers from RAG context or web results."""
    
    def __init__(self, llm: BaseChatModel):
        """
        Initialize answer generator.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
        self.rag_prompt = get_rag_answer_prompt()
        self.web_prompt = get_web_answer_prompt()
        self.reformulation_prompt = get_query_reformulation_prompt()
    
    def generate_rag_answer(
        self,
        question: str,
        context: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer from RAG context.
        
        Args:
            question: User question
            context: Retrieved context
            sources: Source documents metadata
        
        Returns:
            Dictionary with answer and metadata
        """
        app_logger.info("Generating RAG answer")
        
        try:
            with Timer("RAG answer generation"):
                # Format prompt
                messages = self.rag_prompt.format_messages(
                    context=context,
                    question=question
                )
                
                # Generate answer
                response = self.llm.invoke(messages)
                answer = response.content
            
            # Format citations
            citations = format_citations(sources, source_type="pdf")
            
            result = {
                'answer': answer,
                'source_type': 'knowledge_base',
                'sources': sources,
                'citations': citations,
                'success': True,
            }
            
            app_logger.info("RAG answer generated successfully")
            return result
        
        except Exception as e:
            app_logger.error(f"Failed to generate RAG answer: {str(e)}")
            return {
                'answer': "I apologize, but I encountered an error generating the answer.",
                'source_type': 'knowledge_base',
                'sources': [],
                'citations': '',
                'success': False,
                'error': str(e),
            }
    
    def generate_web_answer(
        self,
        question: str,
        web_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer from web search results.
        
        Args:
            question: User question
            web_results: Web search results
        
        Returns:
            Dictionary with answer and metadata
        """
        app_logger.info("Generating web answer")
        
        try:
            # Format web results for prompt
            web_results_text = self._format_web_results(web_results)
            
            with Timer("Web answer generation"):
                # Format prompt
                messages = self.web_prompt.format_messages(
                    web_results=web_results_text,
                    question=question
                )
                
                # Generate answer
                response = self.llm.invoke(messages)
                answer = response.content
            
            # Format citations
            sources = [
                {
                    'url': result['url'],
                    'title': result['title']
                }
                for result in web_results[:5]  # Top 5 sources
            ]
            citations = format_citations(sources, source_type="web")
            
            result = {
                'answer': answer,
                'source_type': 'web',
                'sources': sources,
                'citations': citations,
                'success': True,
            }
            
            app_logger.info("Web answer generated successfully")
            return result
        
        except Exception as e:
            app_logger.error(f"Failed to generate web answer: {str(e)}")
            return {
                'answer': "I apologize, but I encountered an error generating the answer from web sources.",
                'source_type': 'web',
                'sources': [],
                'citations': '',
                'success': False,
                'error': str(e),
            }
    
    def reformulate_query_for_web(self, question: str) -> str:
        """
        Reformulate user question for better web search.
        
        Args:
            question: Original question
        
        Returns:
            Reformulated search query
        """
        try:
            messages = self.reformulation_prompt.format_messages(question=question)
            response = self.llm.invoke(messages)
            reformulated = response.content.strip()
            
            # Remove quotes if they exist (sometimes LLM adds them)
            reformulated = reformulated.strip('"\'')
            
            # If reformulation is too short or seems broken, use simple fallback
            if len(reformulated) < 3 or not reformulated.replace(' ', '').isalnum():
                reformulated = self._simple_reformulation(question)
            
            app_logger.info(f"Query reformulated: '{question}' -> '{reformulated}'")
            return reformulated
        
        except Exception as e:
            app_logger.warning(f"Query reformulation failed: {str(e)}")
            # Fallback to simple reformulation
            return self._simple_reformulation(question)
    
    def _simple_reformulation(self, question: str) -> str:
        """Simple fallback reformulation without LLM."""
        # Remove question words and punctuation
        import re
        query = re.sub(r'^(what|how|when|where|why|who|which|can you|please)\s+', '', question.lower())
        query = re.sub(r'\?$', '', query)
        return query.strip()
    
    def _format_web_results(self, web_results: List[Dict[str, Any]]) -> str:
        """
        Format web results for the prompt.
        
        Args:
            web_results: List of web search results
        
        Returns:
            Formatted string
        """
        formatted = []
        for i, result in enumerate(web_results, 1):
            title = result.get('title', 'No Title')
            url = result.get('url', '')
            content = result.get('content', result.get('snippet', ''))
            
            formatted.append(
                f"[{i}] {title}\n"
                f"URL: {url}\n"
                f"Content: {content[:500]}...\n"
            )
        
        return "\n".join(formatted)
    
    def format_final_response(
        self,
        answer_data: Dict[str, Any],
        include_metadata: bool = True
    ) -> str:
        """
        Format the final response for display.
        
        Args:
            answer_data: Answer data dictionary
            include_metadata: Whether to include source metadata
        
        Returns:
            Formatted response string
        """
        parts = []
        
        # Add answer
        parts.append("**Answer:**")
        parts.append(answer_data['answer'])
        parts.append("")
        
        # Add source information
        if include_metadata and answer_data.get('citations'):
            parts.append("**Sources:**")
            parts.append(answer_data['citations'])
        
        return "\n".join(parts)
