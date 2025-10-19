"""
PDF document loader and processor.
Handles PDF loading, text extraction, and chunking with metadata preservation.
"""

from pathlib import Path
from typing import List, Dict, Any
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config.settings import settings
from src.utils.logger import app_logger
from src.utils.helpers import Timer


class PDFLoader:
    """Handles PDF document loading and processing."""
    
    def __init__(self, pdf_path: Path):
        """
        Initialize PDF loader.
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )
    
    def load_pdf(self) -> List[Document]:
        """
        Load PDF and extract text with metadata.
        
        Returns:
            List of Document objects with page-level metadata
        
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF loading fails
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        app_logger.info(f"Loading PDF: {self.pdf_path}")
        documents = []
        
        try:
            with Timer("PDF loading"):
                with pdfplumber.open(self.pdf_path) as pdf:
                    total_pages = len(pdf.pages)
                    app_logger.info(f"PDF has {total_pages} pages")
                    
                    for page_num, page in enumerate(pdf.pages, start=1):
                        # Extract text from page
                        text = page.extract_text()
                        
                        if text and text.strip():
                            # Create document with metadata
                            doc = Document(
                                page_content=text,
                                metadata={
                                    "source": str(self.pdf_path.name),
                                    "page": page_num,
                                    "total_pages": total_pages,
                                }
                            )
                            documents.append(doc)
                        else:
                            app_logger.warning(f"Page {page_num} contains no extractable text")
            
            app_logger.info(f"Successfully loaded {len(documents)} pages from PDF")
            return documents
        
        except Exception as e:
            app_logger.error(f"Failed to load PDF: {str(e)}")
            raise
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving metadata.
        
        Args:
            documents: List of Document objects
        
        Returns:
            List of chunked Document objects
        """
        app_logger.info(f"Chunking {len(documents)} documents")
        
        try:
            with Timer("Document chunking"):
                chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk index to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
            
            app_logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            
            # Log statistics
            avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
            app_logger.info(f"Average chunk size: {avg_chunk_size:.0f} characters")
            
            return chunks
        
        except Exception as e:
            app_logger.error(f"Failed to chunk documents: {str(e)}")
            raise
    
    def process_pdf(self) -> List[Document]:
        """
        Complete pipeline: load PDF and create chunks.
        
        Returns:
            List of chunked Document objects ready for embedding
        """
        documents = self.load_pdf()
        chunks = self.chunk_documents(documents)
        return chunks
    
    def get_pdf_info(self) -> Dict[str, Any]:
        """
        Get metadata about the PDF.
        
        Returns:
            Dictionary with PDF metadata
        """
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                return {
                    "filename": self.pdf_path.name,
                    "path": str(self.pdf_path),
                    "num_pages": len(pdf.pages),
                    "metadata": pdf.metadata,
                }
        except Exception as e:
            app_logger.error(f"Failed to get PDF info: {str(e)}")
            return {}


def load_and_process_pdf(pdf_path: Path) -> List[Document]:
    """
    Convenience function to load and process a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of processed document chunks
    """
    loader = PDFLoader(pdf_path)
    return loader.process_pdf()
