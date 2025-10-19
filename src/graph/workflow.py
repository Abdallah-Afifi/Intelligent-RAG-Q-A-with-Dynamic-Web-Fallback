"""
LangGraph workflow orchestration.
Defines the graph structure and execution flow.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from src.graph.state import GraphState, create_initial_state
from src.graph.nodes import WorkflowNodes
from src.rag.retriever import RAGRetriever
from src.rag.answer_generator import AnswerGenerator
from src.web_search.web_searcher import WebSearcher
from src.utils.logger import app_logger


class QAWorkflow:
    """Orchestrates the Q&A workflow using LangGraph."""
    
    def __init__(
        self,
        retriever: RAGRetriever,
        answer_generator: AnswerGenerator,
        web_searcher: WebSearcher
    ):
        """
        Initialize Q&A workflow.
        
        Args:
            retriever: RAG retriever instance
            answer_generator: Answer generator instance
            web_searcher: Web searcher instance
        """
        self.nodes = WorkflowNodes(retriever, answer_generator, web_searcher)
        self.graph = self._build_graph()
        app_logger.info("Q&A workflow initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        app_logger.info("Building workflow graph")
        
        # Create graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve_kb", self.nodes.retrieve_from_kb)
        workflow.add_node("generate_rag_answer", self.nodes.generate_rag_answer)
        workflow.add_node("notify_fallback", self.nodes.notify_user_fallback)
        workflow.add_node("search_web", self.nodes.search_web)
        workflow.add_node("generate_web_answer", self.nodes.generate_web_answer)
        workflow.add_node("format_output", self.nodes.format_final_output)
        
        # Set entry point
        workflow.set_entry_point("retrieve_kb")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "retrieve_kb",
            self._route_after_retrieval,
            {
                "generate_rag": "generate_rag_answer",
                "fallback_to_web": "notify_fallback",
            }
        )
        
        # RAG path with potential fallback
        workflow.add_conditional_edges(
            "generate_rag_answer",
            self._route_after_rag_answer,
            {
                "use_rag": "format_output",
                "fallback_to_web": "notify_fallback",
            }
        )
        
        # Web fallback path
        workflow.add_edge("notify_fallback", "search_web")
        workflow.add_edge("search_web", "generate_web_answer")
        workflow.add_edge("generate_web_answer", "format_output")
        
        # End
        workflow.add_edge("format_output", END)
        
        # Compile graph
        app = workflow.compile()
        app_logger.info("Workflow graph compiled successfully")
        
        return app
    
    def _route_after_retrieval(
        self, state: GraphState
    ) -> Literal["generate_rag", "fallback_to_web"]:
        """
        Routing function to decide between RAG and web search.
        
        Args:
            state: Current graph state
        
        Returns:
            Next node to execute
        """
        if state['is_knowledge_base_sufficient']:
            app_logger.info("ROUTING: Using knowledge base (RAG)")
            return "generate_rag"
        else:
            app_logger.info("ROUTING: Falling back to web search")
            return "fallback_to_web"
    
    def _route_after_rag_answer(
        self, state: GraphState
    ) -> Literal["use_rag", "fallback_to_web"]:
        """
        Routing function after RAG answer generation.
        Checks if the answer indicates insufficient information.
        
        Args:
            state: Current graph state
        
        Returns:
            Next node to execute
        """
        if state.get('needs_web_fallback', False):
            app_logger.info("ROUTING: RAG answer insufficient, falling back to web")
            return "fallback_to_web"
        else:
            app_logger.info("ROUTING: Using RAG answer")
            return "use_rag"
    
    def run(self, question: str) -> GraphState:
        """
        Run the Q&A workflow for a question.
        
        Args:
            question: User's question
        
        Returns:
            Final state with answer
        """
        app_logger.info(f"Running workflow for question: {question}")
        
        # Create initial state
        initial_state = create_initial_state(question)
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            app_logger.info(f"Workflow completed successfully (source: {final_state['source_type']})")
            return final_state
        
        except Exception as e:
            app_logger.error(f"Workflow execution failed: {str(e)}")
            initial_state['error'] = str(e)
            initial_state['answer'] = "An error occurred while processing your question."
            return initial_state
    
    def get_graph_visualization(self) -> str:
        """
        Get a text representation of the graph structure.
        
        Returns:
            Graph visualization string
        """
        return """
Q&A Workflow Graph:

START
  ↓
[retrieve_kb] ← Retrieve from knowledge base
  ↓
  ├─(sufficient)─→ [generate_rag_answer] ← Generate answer from KB
  │                   ↓
  │                [format_output]
  │                   ↓
  │                 END
  │
  └─(insufficient)─→ [notify_fallback] ← Notify user
                        ↓
                    [search_web] ← Search the web
                        ↓
                    [generate_web_answer] ← Generate from web
                        ↓
                    [format_output]
                        ↓
                      END
        """


def create_workflow(
    retriever: RAGRetriever,
    answer_generator: AnswerGenerator,
    web_searcher: WebSearcher
) -> QAWorkflow:
    """
    Factory function to create a Q&A workflow.
    
    Args:
        retriever: RAG retriever instance
        answer_generator: Answer generator instance
        web_searcher: Web searcher instance
    
    Returns:
        QAWorkflow instance
    """
    return QAWorkflow(retriever, answer_generator, web_searcher)
