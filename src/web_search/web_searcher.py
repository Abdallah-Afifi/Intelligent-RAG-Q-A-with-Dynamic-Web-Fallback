"""
Web search implementation using DuckDuckGo + requests + BeautifulSoup.
Provides a generic, broad web search via DuckDuckGo and keeps
lightweight scrapers for Wikipedia/StackOverflow as fallbacks, plus a
special-case provider for current weather.
"""

from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import urllib.parse
import random
import time
import re
from config.settings import settings
from src.utils.logger import app_logger
from src.utils.helpers import Timer, retry_on_failure
from duckduckgo_search import DDGS


class WebSearcher:
    """Handles web search operations using requests + BeautifulSoup."""
    
    def __init__(self):
        """Initialize web searcher."""
        self.max_results = settings.MAX_SEARCH_RESULTS
        self.timeout = settings.WEB_SEARCH_TIMEOUT
        
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
    
    def _get_headers(self) -> Dict[str, str]:
        """Get randomized headers to avoid detection."""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform a generic search with DuckDuckGo, with special-case routing:
        - Weather queries: wttr.in
        - Generic web search: DuckDuckGo (broad coverage)
        - Fallbacks: Wikipedia and StackOverflow scraping if needed
        
        Args:
            query: Search query
            max_results: Maximum number of results (defaults to config value)
        
        Returns:
            List of search result dictionaries: {title, url, snippet}
        """
        max_results = max_results or self.max_results
        app_logger.info(f"Searching web for: {query}")

        with Timer("Web search"):
            # 1) Weather intent detection
            weather_match = re.search(r"current\s+weather\s+in\s+(.+)", query, re.IGNORECASE)
            if weather_match:
                city = weather_match.group(1).strip()
                results = self._search_weather(city)
                if results:
                    return results[:max_results]

            # 2) Generic DuckDuckGo search (broad coverage)
            ddg_results = self._search_duckduckgo(query, max_results)
            if ddg_results:
                return ddg_results[:max_results]

            # 3) Wikipedia scraping (general knowledge fallback)
            wiki_results = self._search_wikipedia(query, max_results)
            if wiki_results:
                return wiki_results[:max_results]

            # 4) StackOverflow scraping (technical queries fallback)
            so_results = self._search_stackoverflow(query, max_results)
            if so_results:
                return so_results[:max_results]

        
        app_logger.info("Found 0 search results")
        return []

    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generic web search using DuckDuckGo (no API key required)."""
        results: List[Dict[str, Any]] = []
        try:
            
            time.sleep(random.uniform(0.2, 0.6))
            with DDGS() as ddgs:
                
                for r in ddgs.text(query, max_results=max_results, safesearch="moderate"):  # type: ignore[arg-type]
                    title = r.get("title") or "No Title"
                    url = r.get("href") or ""
                    snippet = r.get("body") or ""
                    if not url:
                        continue
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    })
        except Exception as e:
            app_logger.debug(f"DuckDuckGo search failed: {e}")
        return results

    def _search_wikipedia(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Scrape Wikipedia search results page for the query."""
        try:
            encoded = urllib.parse.quote_plus(query)
            url = (
                "https://en.wikipedia.org/w/index.php?search="
                f"{encoded}&title=Special:Search&profile=advanced&fulltext=1&ns0=1"
            )
            time.sleep(random.uniform(0.3, 0.8))
            resp = requests.get(url, headers=self._get_headers(), timeout=self.timeout)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')
            results: List[Dict[str, Any]] = []

            # Result items: div.mw-search-result-heading > a
            for a in soup.select('div.mw-search-result-heading > a')[:max_results]:
                href = a.get('href')
                if not href:
                    continue
                title = a.get_text(strip=True)
                full_url = urllib.parse.urljoin('https://en.wikipedia.org', href)

                # Snippet
                snippet_div = a.find_parent('div').find_next_sibling('div')
                snippet = snippet_div.get_text(" ", strip=True) if snippet_div else ''

                results.append({
                    'title': title or 'No Title',
                    'url': full_url,
                    'snippet': snippet or 'Wikipedia result',
                })

            # If no results from search page, try direct page summary pattern
            if not results:
                # Try exact page
                page_url = 'https://en.wikipedia.org/wiki/' + encoded.replace('%20', '_')
                resp2 = requests.get(page_url, headers=self._get_headers(), timeout=self.timeout)
                if resp2.status_code == 200 and 'Wikipedia' in resp2.text:
                    soup2 = BeautifulSoup(resp2.text, 'html.parser')
                    title_tag = soup2.select_one('#firstHeading')
                    title = title_tag.get_text(strip=True) if title_tag else query
                    # First paragraph
                    para = soup2.select_one('div.mw-parser-output > p')
                    snippet = para.get_text(" ", strip=True) if para else ''
                    results.append({
                        'title': title,
                        'url': page_url,
                        'snippet': snippet or 'Wikipedia article',
                    })

            return results
        except Exception as e:
            app_logger.debug(f"Wikipedia scrape failed: {e}")
            return []

    def _search_stackoverflow(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Scrape StackOverflow search results for the query."""
        try:
            encoded = urllib.parse.quote_plus(query)
            url = f"https://stackoverflow.com/search?q={encoded}"
            time.sleep(random.uniform(0.3, 0.8))
            resp = requests.get(url, headers=self._get_headers(), timeout=self.timeout)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')
            results: List[Dict[str, Any]] = []

            # New SO layout uses a list of question summaries
            for q in soup.select('div.s-result--content')[:max_results]:
                a = q.select_one('a.s-link')
                if not a:
                    continue
                title = a.get_text(strip=True)
                href = a.get('href', '')
                full_url = urllib.parse.urljoin('https://stackoverflow.com', href)
                snippet_el = q.select_one('div.s-result--snippet') or q.select_one('div.fc-medium')
                snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ''
                results.append({
                    'title': title or 'No Title',
                    'url': full_url,
                    'snippet': snippet or 'StackOverflow result',
                })

            return results
        except Exception as e:
            app_logger.debug(f"StackOverflow scrape failed: {e}")
            return []

    def _search_weather(self, city: str) -> List[Dict[str, Any]]:
        """Fetch current weather using wttr.in (no API key, plain text)."""
        try:
            city_encoded = urllib.parse.quote_plus(city)
            # Compact one-line weather
            url_txt = f"https://wttr.in/{city_encoded}?format=3"
            time.sleep(random.uniform(0.2, 0.5))
            resp_txt = requests.get(url_txt, headers=self._get_headers(), timeout=self.timeout)
            if resp_txt.status_code == 200 and resp_txt.text.strip():
                line = resp_txt.text.strip()
                return [{
                    'title': f"Current weather in {city}",
                    'url': f"https://wttr.in/{city_encoded}",
                    'snippet': line,
                }]
        except Exception as e:
            app_logger.debug(f"Weather fetch failed: {e}")
        return []
    
    @retry_on_failure(max_retries=2, delay=1.0)
    def fetch_page_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract main content from a webpage.
        
        Args:
            url: URL of the webpage
        
        Returns:
            Extracted text content or None if failed
        """
        app_logger.debug(f"Fetching content from: {url}")
        
        try:
            # Add small delay to be respectful
            time.sleep(random.uniform(0.3, 0.8))
            
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
            
            # Limit content length
            if len(text) > settings.MAX_WEB_CONTENT_LENGTH:
                text = text[:settings.MAX_WEB_CONTENT_LENGTH]
                app_logger.debug(f"Content truncated to {settings.MAX_WEB_CONTENT_LENGTH} characters")
            
            app_logger.debug(f"Extracted {len(text)} characters from {url}")
            return text
        
        except requests.exceptions.RequestException as e:
            app_logger.warning(f"Failed to fetch {url}: {str(e)}")
            return None
        except Exception as e:
            app_logger.warning(f"Error processing {url}: {str(e)}")
            return None
    
    def search_and_extract(
        self,
        query: str,
        max_results: Optional[int] = None,
        extract_content: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search and optionally extract full content from results.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            extract_content: Whether to fetch full page content
        
        Returns:
            List of result dictionaries with content
        """
        results = self.search(query, max_results)
        
        if not extract_content:
            return results
        
        # Fetch content for each result
        enriched_results = []
        for result in results:
            url = result.get('url', '')
            if url:
                content = self.fetch_page_content(url)
                if content:
                    result['content'] = content
                else:
                    result['content'] = result['snippet']  # Fallback to snippet
            else:
                result['content'] = result['snippet']
            
            enriched_results.append(result)
        
        return enriched_results
    
    def reformulate_query(self, original_query: str, context: str = "") -> str:
        """
        Reformulate query for better search results.
        
        Args:
            original_query: Original user query
            context: Additional context
        
        Returns:
            Reformulated query
        """
        # Simple reformulation - can be enhanced with LLM
        query = original_query.strip()
        
        # Remove question marks and make more search-friendly
        query = query.replace('?', '')
        
        # Add context if available
        if context:
            query = f"{query} {context}"
        
        app_logger.debug(f"Reformulated query: '{original_query}' -> '{query}'")
        return query


def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function for web search.
    
    Args:
        query: Search query
        max_results: Maximum number of results
    
    Returns:
        List of search results
    """
    searcher = WebSearcher()
    return searcher.search_and_extract(query, max_results)
