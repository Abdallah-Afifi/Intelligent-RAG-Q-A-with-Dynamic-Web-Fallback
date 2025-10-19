"""
LLM Factory for creating language model instances.
Supports multiple free LLM providers: Groq, Ollama, and HuggingFace.
"""

from typing import Optional
from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from config.settings import settings
from src.utils.logger import app_logger


class LLMFactory:
    """Factory class for creating LLM instances."""
    
    @staticmethod
    def create_llm() -> BaseChatModel:
        """
        Create an LLM instance based on configuration.
        
        Returns:
            Configured LLM instance
        
        Raises:
            ValueError: If provider is not supported or configuration is invalid
        """
        provider = settings.LLM_PROVIDER.lower()
        
        app_logger.info(f"Initializing LLM with provider: {provider}, model: {settings.LLM_MODEL}")
        
        try:
            if provider == "groq":
                return LLMFactory._create_groq_llm()
            elif provider == "ollama":
                return LLMFactory._create_ollama_llm()
            elif provider == "huggingface":
                return LLMFactory._create_huggingface_llm()
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        
        except Exception as e:
            app_logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    @staticmethod
    def _create_groq_llm() -> BaseChatModel:
        """Create Groq LLM instance."""
        try:
            from langchain_groq import ChatGroq
            
            if not settings.GROQ_API_KEY:
                raise ValueError(
                    "GROQ_API_KEY not set. Get your free API key from https://console.groq.com"
                )
            
            llm = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            
            app_logger.info("Groq LLM initialized successfully")
            return llm
        
        except ImportError:
            raise ImportError("langchain-groq not installed. Run: pip install langchain-groq")
    
    @staticmethod
    def _create_ollama_llm() -> BaseChatModel:
        """Create Ollama LLM instance."""
        try:
            from langchain_community.chat_models import ChatOllama
            
            llm = ChatOllama(
                model=settings.LLM_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=settings.LLM_TEMPERATURE,
            )
            
            app_logger.info(f"Ollama LLM initialized successfully with model: {settings.LLM_MODEL}")
            app_logger.info(f"Make sure Ollama is running and model is downloaded: ollama pull {settings.LLM_MODEL}")
            return llm
        
        except ImportError:
            raise ImportError("Required Ollama dependencies not installed")
    
    @staticmethod
    def _create_huggingface_llm() -> BaseChatModel:
        """Create HuggingFace LLM instance."""
        try:
            from langchain_community.chat_models import ChatHuggingFace
            from langchain_community.llms import HuggingFaceHub
            
            if not settings.HUGGINGFACE_API_KEY:
                raise ValueError(
                    "HUGGINGFACE_API_KEY not set. Get your free API key from https://huggingface.co/settings/tokens"
                )
            
            llm = HuggingFaceHub(
                repo_id=settings.LLM_MODEL,
                huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
                model_kwargs={
                    "temperature": settings.LLM_TEMPERATURE,
                    "max_length": settings.LLM_MAX_TOKENS,
                }
            )
            
            app_logger.info("HuggingFace LLM initialized successfully")
            return llm
        
        except ImportError:
            raise ImportError("Required HuggingFace dependencies not installed")
    
    @staticmethod
    def test_llm(llm: BaseChatModel) -> bool:
        """
        Test if LLM is working properly.
        
        Args:
            llm: LLM instance to test
        
        Returns:
            True if test successful, False otherwise
        """
        try:
            app_logger.info("Testing LLM connection...")
            
            # Use invoke instead of __call__ for BaseChatModel
            from langchain_core.messages import HumanMessage
            test_message = [HumanMessage(content="Say 'Hello' if you can read this.")]
            response = llm.invoke(test_message)
            
            app_logger.info(f"LLM test successful. Response: {response.content[:50]}...")
            return True
        
        except Exception as e:
            app_logger.error(f"LLM test failed: {str(e)}")
            return False


def get_llm() -> BaseChatModel:
    """
    Convenience function to get a configured LLM instance.
    
    Returns:
        Configured LLM instance
    """
    return LLMFactory.create_llm()
