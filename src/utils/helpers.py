"""Helper utilities for the RAG Q&A system."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
import pickle
from functools import wraps
import time
from src.utils.logger import app_logger


def hash_text(text: str) -> str:
    """Generate a hash for text content."""
    return hashlib.md5(text.encode()).hexdigest()


def cache_result(cache_dir: Path, cache_key: str, ttl_seconds: int = 3600):
    """
    Decorator to cache function results to disk.
    
    Args:
        cache_dir: Directory to store cache files
        cache_key: Unique identifier for the cache
        ttl_seconds: Time to live in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate cache filename
            args_hash = hash_text(str(args) + str(kwargs))
            cache_file = cache_dir / f"{cache_key}_{args_hash}.pkl"
            
            # Check if cache exists and is valid
            if cache_file.exists():
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < ttl_seconds:
                    try:
                        with open(cache_file, 'rb') as f:
                            app_logger.debug(f"Cache hit for {cache_key}")
                            return pickle.load(f)
                    except Exception as e:
                        app_logger.warning(f"Cache read error: {e}")
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                app_logger.debug(f"Cache stored for {cache_key}")
            except Exception as e:
                app_logger.warning(f"Cache write error: {e}")
            
            return result
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry a function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    app_logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {str(e)}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
            
            app_logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator


def format_citations(sources: list, source_type: str = "pdf") -> str:
    """
    Format citations for display.
    
    Args:
        sources: List of source documents or URLs
        source_type: Type of source ("pdf" or "web")
    
    Returns:
        Formatted citation string
    """
    if not sources:
        return "No sources available."
    
    citations = []
    if source_type == "pdf":
        for i, source in enumerate(sources, 1):
            page = source.get('page', 'Unknown')
            citations.append(f"[{i}] Page {page}")
    else:  # web
        for i, source in enumerate(sources, 1):
            url = source.get('url', 'Unknown')
            title = source.get('title', 'Untitled')
            citations.append(f"[{i}] {title} - {url}")
    
    return "\n".join(citations)


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def calculate_relevance_score(similarity_scores: list) -> float:
    """
    Calculate overall relevance score from individual similarity scores.
    
    Args:
        similarity_scores: List of similarity scores
    
    Returns:
        Overall relevance score between 0 and 1
    """
    if not similarity_scores:
        return 0.0
    
    # Use weighted average with exponential decay for lower-ranked results
    weights = [1.0 / (i + 1) for i in range(len(similarity_scores))]
    weighted_sum = sum(score * weight for score, weight in zip(similarity_scores, weights))
    weight_total = sum(weights)
    
    return weighted_sum / weight_total if weight_total > 0 else 0.0


def save_json(data: Dict[Any, Any], filepath: Path):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> Dict[Any, Any]:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        app_logger.info(f"{self.name} took {duration:.2f} seconds")
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration if timing is complete."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
