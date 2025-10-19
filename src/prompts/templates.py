"""
Prompt templates for the RAG Q&A system.
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


# RAG Answer Generation Prompt
RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable assistant that answers questions based on provided context.

Your task:
1. Answer the user's question using ONLY the information from the provided context
2. Be accurate and cite specific pages when possible
3. If the context doesn't contain enough information to fully answer, say so
4. Do not make up information or use external knowledge
5. Be concise but complete

Format your answer clearly with proper citations."""),
    ("human", """Context from knowledge base:
{context}

Question: {question}

Please provide a detailed answer based on the context above. Include page references in your answer.""")
])


# Web Answer Generation Prompt
WEB_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that synthesizes information from web sources.

Your task:
1. Answer the question using the web search results provided
2. Combine information from multiple sources when relevant
3. Be accurate and mention which sources support each claim
4. If information is conflicting, acknowledge different perspectives
5. Provide a clear, well-structured answer

Always cite your sources by referencing them as [1], [2], etc."""),
    ("human", """Web search results:
{web_results}

Question: {question}

Please provide a comprehensive answer based on the web results above. Cite sources appropriately.""")
])


# Query Analysis Prompt
QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at analyzing user queries.

Your task:
1. Identify the main topic and intent of the question
2. Extract key concepts that would be useful for search
3. Determine if this is a factual question that can be answered with documents
4. Suggest search keywords if needed

Be concise and analytical."""),
    ("human", """Analyze this question:
{question}

Provide:
- Main topic
- Key concepts
- Question type (factual, opinion, how-to, etc.)
- Suggested search keywords""")
])


# Relevance Evaluation Prompt (for LLM-based assessment)
RELEVANCE_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at evaluating if provided context can answer a question.

Your task:
Rate from 0-10 how well the context can answer the question:
- 0-3: Context is irrelevant or unhelpful
- 4-6: Context is somewhat related but insufficient
- 7-8: Context can partially answer the question
- 9-10: Context fully answers the question

Be strict and honest in your evaluation."""),
    ("human", """Question: {question}

Context:
{context}

Rate the relevance (0-10) and explain briefly why.""")
])


# Query Reformulation Prompt
QUERY_REFORMULATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at reformulating questions for better web search results.

Your task:
Transform the user's question into an optimal search query:
- Remove unnecessary words
- Use keywords that would appear in relevant documents
- Keep it concise (3-7 words typically)
- Focus on the core information need

Return ONLY the search query, nothing else."""),
    ("human", """Original question: {question}

Reformulated search query:""")
])


# Source Citation Formatting Prompt
CITATION_FORMAT_PROMPT = """
Format the following sources as proper citations:

{sources}

Return formatted citations ready for display.
"""


def get_rag_answer_prompt() -> ChatPromptTemplate:
    """Get the RAG answer generation prompt."""
    return RAG_ANSWER_PROMPT


def get_web_answer_prompt() -> ChatPromptTemplate:
    """Get the web answer generation prompt."""
    return WEB_ANSWER_PROMPT


def get_query_analysis_prompt() -> ChatPromptTemplate:
    """Get the query analysis prompt."""
    return QUERY_ANALYSIS_PROMPT


def get_relevance_eval_prompt() -> ChatPromptTemplate:
    """Get the relevance evaluation prompt."""
    return RELEVANCE_EVAL_PROMPT


def get_query_reformulation_prompt() -> ChatPromptTemplate:
    """Get the query reformulation prompt."""
    return QUERY_REFORMULATION_PROMPT
