"""Main Q&A System with RAG and Web Fallback."""

from pathlib import Path
from typing import Dict, Any, Optional
from src.llm.llm_factory import get_llm
from src.embeddings.embedding_factory import get_embeddings
from src.document_processing.pdf_loader import load_and_process_pdf
from src.vector_store.chroma_store import ChromaVectorStore, get_or_create_vectorstore
from src.rag.retriever import RAGRetriever
from src.rag.answer_generator import AnswerGenerator
from src.web_search.web_searcher import WebSearcher
from src.graph.workflow import create_workflow, QAWorkflow
from src.utils.logger import app_logger
from config.settings import settings


class QASystem:
    """Main Q&A System with RAG and Web Fallback capability."""
    
    def __init__(self, knowledge_base_path: Optional[Path] = None):
        """Initialize the Q&A system."""
        self.knowledge_base_path = knowledge_base_path or settings.KNOWLEDGE_BASE_PATH
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.answer_generator = None
        self.web_searcher = None
        self.workflow = None
        
        app_logger.info("Initializing Q&A System")
    
    def setup(self, force_reload_kb: bool = False) -> bool:
        """
        Set up all components of the system.
        
        Args:
            force_reload_kb: Whether to force reload the knowledge base
        
        Returns:
            True if setup successful
        """
        try:
            app_logger.info("Setting up Q&A System components...")
            
            # 1. Initialize LLM
            app_logger.info("Initializing LLM...")
            self.llm = get_llm()
            
            # 2. Initialize embeddings
            app_logger.info("Initializing embeddings...")
            self.embeddings = get_embeddings()
            
            # 3. Load and process knowledge base
            app_logger.info("Loading knowledge base...")
            self.vector_store = self._setup_knowledge_base(force_reload_kb)
            
            # 4. Initialize RAG retriever
            app_logger.info("Initializing RAG retriever...")
            self.retriever = RAGRetriever(self.vector_store)
            
            # 5. Initialize answer generator
            app_logger.info("Initializing answer generator...")
            self.answer_generator = AnswerGenerator(self.llm)
            
            # 6. Initialize web searcher
            app_logger.info("Initializing web searcher...")
            self.web_searcher = WebSearcher()
            
            # 7. Create workflow
            app_logger.info("Creating workflow...")
            self.workflow = create_workflow(
                retriever=self.retriever,
                answer_generator=self.answer_generator,
                web_searcher=self.web_searcher
            )
            
            app_logger.info("Q&A System setup complete!")
            return True
        
        except Exception as e:
            app_logger.error(f"Failed to setup Q&A System: {str(e)}")
            raise
    
    def _setup_knowledge_base(self, force_reload: bool = False) -> ChromaVectorStore:
        """
        Set up the knowledge base vector store.
        
        Args:
            force_reload: Whether to force reload
        
        Returns:
            ChromaVectorStore instance
        """
        # Check if knowledge base exists
        if not self.knowledge_base_path.exists():
            app_logger.warning(
                f"Knowledge base not found at: {self.knowledge_base_path}\n"
                "The system will work in web-only mode."
            )
            # Create empty vector store
            vector_store = ChromaVectorStore(self.embeddings)
            return vector_store
        
        # Load or create vector store
        vector_store = ChromaVectorStore(self.embeddings)
        
        if vector_store.exists() and not force_reload:
            app_logger.info("Loading existing vector store...")
            vector_store.load_vectorstore()
        else:
            app_logger.info("Processing PDF and creating vector store...")
            documents = load_and_process_pdf(self.knowledge_base_path)
            vector_store.create_vectorstore(documents, force_recreate=force_reload)
        
        # Log statistics
        stats = vector_store.get_collection_stats()
        app_logger.info(f"Vector store stats: {stats}")
        
        return vector_store
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the system.
        
        Args:
            question: User's question
        
        Returns:
            Dictionary with answer and metadata
        """
        if not self.workflow:
            raise RuntimeError("System not initialized. Call setup() first.")
        
        app_logger.info(f"Processing question: {question}")
        
        # Run workflow
        result = self.workflow.run(question)
        
        # Format response
        response = {
            'question': question,
            'answer': result['answer'],
            'source_type': result['source_type'],
            'citations': result['citations'],
            'metadata': result['metadata'],
        }
        
        # Add fallback notification if present
        if 'fallback_notification' in result['metadata']:
            response['fallback_notification'] = result['metadata']['fallback_notification']
        
        return response
    
    def display_response(self, response: Dict[str, Any]):
        """
        Display response in a formatted way.
        
        Args:
            response: Response dictionary from ask()
        """
        print("\n" + "=" * 80)
        print("QUESTION:")
        print("-" * 80)
        print(response['question'])
        print()
        
        # Show fallback notification if present
        if 'fallback_notification' in response:
            print("NOTE:")
            print("-" * 80)
            print(response['fallback_notification'])
            print()
        
        print("ANSWER:")
        print("-" * 80)
        print(response['answer'])
        print()
        
        if response.get('citations'):
            print("SOURCES:")
            print("-" * 80)
            print(response['citations'])
            print()
        
        print(f"Source Type: {response['source_type']}")
        print("=" * 80 + "\n")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the system configuration.
        
        Returns:
            Dictionary with system information
        """
        info = {
            'llm_provider': settings.LLM_PROVIDER,
            'llm_model': settings.LLM_MODEL,
            'embedding_model': settings.EMBEDDING_MODEL,
            'knowledge_base_path': str(self.knowledge_base_path),
            'knowledge_base_exists': self.knowledge_base_path.exists(),
            'vector_store_stats': self.vector_store.get_collection_stats() if self.vector_store else {},
        }
        return info
    
    def reload_knowledge_base(self, new_path: Optional[Path] = None):
        """
        Reload the knowledge base from a new PDF.
        
        Args:
            new_path: Optional new path to PDF
        """
        if new_path:
            self.knowledge_base_path = new_path
        
        app_logger.info(f"Reloading knowledge base from: {self.knowledge_base_path}")
        self.vector_store = self._setup_knowledge_base(force_reload=True)
        self.retriever = RAGRetriever(self.vector_store)
        
        # Recreate workflow with new retriever
        self.workflow = create_workflow(
            retriever=self.retriever,
            answer_generator=self.answer_generator,
            web_searcher=self.web_searcher
        )
        
        app_logger.info("Knowledge base reloaded successfully")


def main():
    """Main entry point for running the system interactively."""
    print("=" * 80)
    print("Intelligent RAG Q&A with Dynamic Web Fallback")
    print("=" * 80)
    print()
    
    # Initialize system
    qa_system = QASystem()
    
    try:
        # Setup
        print("Setting up system...")
        qa_system.setup()
        print("System ready!")
        print()
        
        # Show system info
        info = qa_system.get_system_info()
        print("System Configuration:")
        print(f"  LLM: {info['llm_provider']} - {info['llm_model']}")
        print(f"  Embeddings: {info['embedding_model']}")
        print(f"  Knowledge Base: {info['knowledge_base_path']}")
        print(f"  KB Status: {'Loaded' if info['knowledge_base_exists'] else 'Not Found'}")
        print()
        
        # Interactive loop
        print("Enter your questions (type 'quit' to exit):")
        print("-" * 80)
        
        while True:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            # Get answer
            response = qa_system.ask(question)
            qa_system.display_response(response)
    
    except Exception as e:
        app_logger.error(f"System error: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Please check the logs for details.")


if __name__ == "__main__":
    main()
