"""
CustomRAG - An Advanced Retrieval Augmented Generation System for Local Document Q&A
----------------------------------------------------------------------------------
This application allows users to query their own document collection using 
natural language. It supports various document formats including PDF, DOCX, 
TXT, PowerPoint, and EPUB files.

Features:
- Document embedding and vector storage using ChromaDB
- Multi-strategy retrieval with hybrid search capabilities
- Advanced query processing and classification
- Smart context generation with dynamic prioritization
- Template-based response generation with citations
- Support for multiple file formats
- Conversation memory with context awareness
- GPU acceleration when available (CUDA/MPS)

Author: Nik
License: MIT
"""

import os
import shutil
import sqlite3
import time
import logging
import torch
import sys
import json
import re
import random
import math
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from pathlib import Path
from collections import Counter, defaultdict
from enum import Enum
from datetime import datetime
from chromadb.config import Settings

# NLP imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.util import ngrams

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# EPUB processing imports
import html2text
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

# Document processing imports
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredEPubLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

# Vector store and embedding imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LLM and prompt imports
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import requests

# Configure logging - production settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("customrag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CustomRAG")

#############################################################################
# CONFIGURATION
#############################################################################

class Config:
    """Configuration settings for the RAG application."""
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.environ.get("CUSTOMRAG_DATA_PATH", os.path.join(SCRIPT_DIR, "data"))
    PERSIST_PATH = os.environ.get("CUSTOMRAG_VECTOR_PATH", os.path.join(SCRIPT_DIR, "vector_store"))
    
    # Embedding and retrieval settings
    EMBEDDING_MODEL_NAME = os.environ.get("CUSTOMRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL_NAME = os.environ.get("CUSTOMRAG_LLM_MODEL", "deepseek-r1:8b")
    
    # Chunking settings
    CHUNK_SIZE = int(os.environ.get("CUSTOMRAG_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.environ.get("CUSTOMRAG_CHUNK_OVERLAP", "150"))
    
    # Retrieval settings
    RETRIEVER_K = int(os.environ.get("CUSTOMRAG_RETRIEVER_K", "10"))
    RETRIEVER_SEARCH_TYPE = os.environ.get("CUSTOMRAG_SEARCH_TYPE", "hybrid")  # Changed to hybrid as default
    KEYWORD_RATIO = float(os.environ.get("CUSTOMRAG_KEYWORD_RATIO", "0.3"))  # 30% weight to keywords by default
    
    # Query processing settings
    USE_QUERY_EXPANSION = os.environ.get("CUSTOMRAG_QUERY_EXPANSION", "True").lower() in ("true", "1", "t")
    EXPANSION_FACTOR = int(os.environ.get("CUSTOMRAG_EXPANSION_FACTOR", "3"))
    
    # Context processing settings
    MAX_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_MAX_CONTEXT_SIZE", "4000"))
    USE_CONTEXT_COMPRESSION = os.environ.get("CUSTOMRAG_CONTEXT_COMPRESSION", "True").lower() in ("true", "1", "t")
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("CUSTOMRAG_OLLAMA_URL", "http://localhost:11434")
    
    # Miscellaneous
    FORCE_REINDEX = os.environ.get("CUSTOMRAG_FORCE_REINDEX", "False").lower() in ("true", "1", "t")
    LOG_LEVEL = os.environ.get("CUSTOMRAG_LOG_LEVEL", "INFO")
    
    # Supported file types
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc', '.md', '.ppt', '.pptx', '.epub']

    @classmethod
    def setup(cls):
        """Set up the configuration and ensure directories exist."""
        # Set logging level based on configuration
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Ensure data directory exists
        os.makedirs(cls.DATA_PATH, exist_ok=True)
        
        logger.info(f"Data directory: {cls.DATA_PATH}")
        logger.info(f"Vector store directory: {cls.PERSIST_PATH}")
        logger.info(f"Embedding model: {cls.EMBEDDING_MODEL_NAME}")
        logger.info(f"LLM model: {cls.LLM_MODEL_NAME}")
        logger.info(f"Search type: {cls.RETRIEVER_SEARCH_TYPE}")
        
        if cls.RETRIEVER_SEARCH_TYPE == "hybrid":
            logger.info(f"Keyword ratio: {cls.KEYWORD_RATIO}")
        
        if cls.USE_QUERY_EXPANSION:
            logger.info(f"Query expansion enabled with factor: {cls.EXPANSION_FACTOR}")
            
        if cls.USE_CONTEXT_COMPRESSION:
            logger.info(f"Context compression enabled")

#############################################################################
# ENUMS AND TYPES
#############################################################################

class QueryType(Enum):
    """Enum for different types of queries."""
    FACTUAL = "factual"           # Asking for specific facts
    PROCEDURAL = "procedural"     # How to do something
    CONCEPTUAL = "conceptual"     # Understanding concepts
    EXPLORATORY = "exploratory"   # Open-ended exploration
    COMPARATIVE = "comparative"   # Comparing things
    CONVERSATIONAL = "conversational"  # Social conversation
    COMMAND = "command"           # System commands
    UNKNOWN = "unknown"           # Unclassified

class DocumentRelevance(Enum):
    """Enum for document relevance levels."""
    HIGH = "high"       # Directly relevant
    MEDIUM = "medium"   # Somewhat relevant
    LOW = "low"         # Tangentially relevant
    NONE = "none"       # Not relevant

class RetrievalStrategy(Enum):
    """Enum for retrieval strategies."""
    SEMANTIC = "semantic"         # Semantic similarity search
    KEYWORD = "keyword"           # Keyword-based search
    HYBRID = "hybrid"             # Combined semantic and keyword
    MMR = "mmr"                   # Maximum Marginal Relevance for diversity

#############################################################################
# 1. INPUT PROCESSING
#############################################################################

class InputProcessor:
    """Processes user input to enhance retrieval quality."""
    
    def __init__(self):
        """Initialize the input processor with NLP components."""
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load common synonyms (simplified version)
        self.synonyms = {
            "document": ["file", "text", "paper", "doc", "content"],
            "find": ["locate", "search", "discover", "retrieve", "get"],
            "explain": ["describe", "clarify", "elaborate", "detail", "elucidate"],
            "show": ["display", "present", "exhibit", "demonstrate"],
            "information": ["info", "data", "details", "facts", "knowledge"],
            "create": ["make", "generate", "produce", "build", "develop"],
            "modify": ["change", "alter", "adjust", "edit", "update"],
            "remove": ["delete", "eliminate", "erase", "take out"],
            "important": ["significant", "essential", "critical", "key", "vital"],
            "problem": ["issue", "challenge", "difficulty", "trouble", "obstacle"],
            "solution": ["answer", "resolution", "fix", "remedy", "workaround"],
        }
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query by converting to lowercase, removing punctuation,
        and standardizing whitespace.
        """
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove punctuation except apostrophes in contractions
        normalized = re.sub(r'[^\w\s\']', ' ', normalized)
        
        # Replace multiple spaces with a single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to extract features for better retrieval.
        
        Returns:
            A dictionary with query analysis including:
            - tokens: List of tokens
            - normalized_query: Normalized query text
            - lemmatized_tokens: Lemmatized tokens
            - keywords: Key terms (non-stopwords)
            - query_type: Type of query (enum)
            - expanded_queries: List of expanded query variants
        """
        # Normalize the query
        normalized_query = self.normalize_query(query)
        
        # Tokenize
        tokens = word_tokenize(normalized_query)
        
        # Remove stopwords and get keywords
        keywords = [token for token in tokens if token.lower() not in self.stop_words and len(token) > 1]
        
        # Lemmatize tokens
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Generate n-grams for phrases (bigrams and trigrams)
        bigrams = list(ngrams(tokens, 2))
        trigrams = list(ngrams(tokens, 3))
        
        # Extract bigram and trigram phrases
        bigram_phrases = [' '.join(bg) for bg in bigrams]
        trigram_phrases = [' '.join(tg) for tg in trigrams]
        
        # Identify query type
        query_type = self.classify_query_type(normalized_query, tokens)
        
        # Generate expanded queries if enabled
        expanded_queries = []
        if Config.USE_QUERY_EXPANSION:
            expanded_queries = self.expand_query(normalized_query, keywords)
        
        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "tokens": tokens,
            "keywords": keywords,
            "lemmatized_tokens": lemmatized_tokens,
            "bigram_phrases": bigram_phrases,
            "trigram_phrases": trigram_phrases,
            "query_type": query_type,
            "expanded_queries": expanded_queries
        }
    
    def classify_query_type(self, query: str, tokens: List[str]) -> QueryType:
        """
        Classify the query into different types.
        """
        # Check for command queries first
        command_patterns = [
            r'\b(?:exit|quit|bye|goodbye)\b',
            r'\bclear\b',
            r'\breindex\b',
            r'\bstats\b',
            r'\bhelp\b'
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.COMMAND
        
        # Check for conversational queries
        conversational_patterns = [
            r'\b(?:hi|hello|hey|greetings|howdy|good\s*(?:morning|afternoon|evening)|what\'s\s*up)\b',
            r'\bhow\s+are\s+you\b',
            r'\b(?:thanks|thank\s*you)\b',
            r'\bappreciate\s*(?:it|that)\b',
        ]
        
        for pattern in conversational_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.CONVERSATIONAL
        
        # Check for factual queries
        factual_patterns = [
            r'\bwhat\s+is\b',
            r'\bwhat\s+are\b',
            r'\bwho\s+is\b',
            r'\bwhen\s+\w+\b',
            r'\bwhere\s+\w+\b',
            r'\blist\b',
            r'\bdefine\b',
            r'\btell\s+me\s+about\b'
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.FACTUAL
        
        # Check for procedural queries
        procedural_patterns = [
            r'\bhow\s+to\b',
            r'\bhow\s+do\s+I\b',
            r'\bhow\s+can\s+I\b',
            r'\bsteps\s+to\b',
            r'\bguide\b',
            r'\btutorial\b',
            r'\bprocedure\b',
            r'\bprocess\b'
        ]
        
        for pattern in procedural_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.PROCEDURAL
        
        # Check for conceptual queries
        conceptual_patterns = [
            r'\bexplain\b',
            r'\bconcept\b',
            r'\btheory\b',
            r'\bwhy\s+\w+\b',
            r'\breasons?\s+for\b',
            r'\bmean\b',
            r'\bunderstand\b'
        ]
        
        for pattern in conceptual_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.CONCEPTUAL
        
        # Check for comparative queries
        comparative_patterns = [
            r'\bcompare\b',
            r'\bcontrast\b',
            r'\bdifference\b',
            r'\bsimilar\b',
            r'\bversus\b',
            r'\bvs\b',
            r'\bbetter\b',
            r'\badvantages?\b',
            r'\bdisadvantages?\b'
        ]
        
        for pattern in comparative_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.COMPARATIVE
        
        # Check for exploratory queries
        exploratory_patterns = [
            r'\btell\s+me\s+more\b',
            r'\blearn\s+about\b',
            r'\bdiscover\b',
            r'\bexplore\b',
            r'\boverview\b',
            r'\bintroduction\b'
        ]
        
        for pattern in exploratory_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.EXPLORATORY
        
        # Default to unknown if no patterns match
        return QueryType.UNKNOWN
    
    def expand_query(self, query: str, keywords: List[str]) -> List[str]:
        """
        Expand the query with synonyms and variations.
        
        Returns:
            List of expanded query strings
        """
        expanded_queries = [query]  # Start with the original query
        
        # Skip expansion for very short queries or conversational queries
        if len(keywords) < 2:
            return expanded_queries
        
        # Expand using synonyms
        for i, keyword in enumerate(keywords):
            if keyword in self.synonyms:
                for synonym in self.synonyms[keyword][:Config.EXPANSION_FACTOR]:
                    # Replace the keyword with its synonym
                    new_keywords = keywords.copy()
                    new_keywords[i] = synonym
                    expanded_query = ' '.join(new_keywords)
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
        
        # If we haven't generated enough variations, try some combinations
        if len(expanded_queries) < Config.EXPANSION_FACTOR and len(keywords) >= 3:
            # Generate variations by removing one non-essential keyword at a time
            for i in range(len(keywords)):
                variation = ' '.join(keywords[:i] + keywords[i+1:])
                if variation not in expanded_queries:
                    expanded_queries.append(variation)
                
                # Stop if we have enough variations
                if len(expanded_queries) >= Config.EXPANSION_FACTOR:
                    break
        
        return expanded_queries[:Config.EXPANSION_FACTOR]  # Limit to avoid too many variations

#############################################################################
# 2. QUERY ROUTING & CLASSIFICATION
#############################################################################

class QueryRouter:
    """Routes queries to appropriate handlers based on query type."""
    
    def __init__(self, conversation_handler, retrieval_handler, command_handler):
        """Initialize with handlers for different query types."""
        self.conversation_handler = conversation_handler
        self.retrieval_handler = retrieval_handler
        self.command_handler = command_handler
    
    def route_query(self, query_analysis: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Route the query to the appropriate handler.
        
        Args:
            query_analysis: Analysis output from InputProcessor
            
        Returns:
            Tuple of (handler_result, should_continue)
        """
        query_type = query_analysis["query_type"]
        original_query = query_analysis["original_query"]
        
        # Route based on query type
        if query_type == QueryType.COMMAND:
            # Handle system commands
            return self.command_handler.handle_command(original_query)
        
        elif query_type == QueryType.CONVERSATIONAL:
            # Handle conversational queries
            response = self.conversation_handler.handle_conversation(original_query)
            return response, True
        
        else:
            # Handle all other query types with retrieval system
            response = self.retrieval_handler.process_query(query_analysis)
            return response, True

class ConversationHandler:
    """Handles conversational queries that don't require document retrieval."""
    
    def __init__(self, memory):
        """Initialize with a memory for conversation history."""
        self.memory = memory
        
        # Greeting patterns and responses
        self.greeting_patterns = [
            r'\b(?:hi|hello|hey|greetings|howdy|good\s*(?:morning|afternoon|evening)|what\'s\s*up)\b',
            r'\bhow\s+are\s+you\b',
        ]

        self.greeting_responses = [
            "Hello! I'm your document assistant. How can I help you with your documents today?",
            "Hi there! I'm ready to help answer questions about your documents. What would you like to know?",
            "Greetings! I'm here to assist with information from your documents. What are you looking for?",
            "Hello! I'm your RAG assistant. I can help you find information in your document collection.",
            "Hi! I'm ready to search your documents. What would you like to learn about?"
        ]

        # Acknowledgement patterns and responses
        self.acknowledgement_patterns = [
            r'\b(?:thanks|thank\s*you)\b',
            r'\bappreciate\s*(?:it|that)\b',
            r'\b(?:awesome|great|cool|nice)\b',
            r'\bthat\s*(?:helps|helped)\b',
            r'\bgot\s*it\b',
        ]

        self.acknowledgement_responses = [
            "You're welcome! Is there anything else you'd like to know about your documents?",
            "Happy to help! Let me know if you have any other questions.",
            "My pleasure! Feel free to ask if you need anything else.",
            "Glad I could assist. Any other questions about your documents?",
            "You're welcome! I'm here if you need more information from your documents."
        ]
    
    def handle_conversation(self, query: str) -> str:
        """
        Handles conversational queries.
        
        Args:
            query: The user's query
            
        Returns:
            Appropriate conversational response
        """
        query_lower = query.lower().strip()
        response = None
        
        # Check for greetings
        for pattern in self.greeting_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                response = random.choice(self.greeting_responses)
                break
        
        # Check for acknowledgements
        if not response:
            for pattern in self.acknowledgement_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    response = random.choice(self.acknowledgement_responses)
                    break
        
        # If no specific pattern matched, give a generic response
        if not response:
            response = "I'm here to help you with your documents. What would you like to know?"
        
        # Update conversation memory
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response)
        
        return response

class CommandHandler:
    """Handles system commands."""
    
    def __init__(self, rag_system):
        """Initialize with a reference to the RAG system."""
        self.rag_system = rag_system
    
    def handle_command(self, command: str) -> Tuple[str, bool]:
        """
        Handles system commands.
        
        Args:
            command: The command string
            
        Returns:
            Tuple of (response, should_continue)
        """
        command_lower = command.lower().strip()
        
        # Help command
        if command_lower in ["help", "menu", "commands"]:
            help_text = """
            Available Commands:
            - help: Display this help menu
            - exit, quit: Stop the application
            - clear: Reset the conversation memory
            - reindex: Reindex all documents
            - stats: See document statistics
            """
            return help_text, True
        
        # Exit commands
        elif command_lower in ["exit", "quit", "bye", "goodbye"]:
            return "Goodbye! Have a great day!", False
            
        # Clear memory command
        elif command_lower == "clear":
            # Reset memory but keep system message
            system_message = self.rag_system.memory.chat_memory.messages[0] if self.rag_system.memory.chat_memory.messages else None
            self.rag_system.memory.clear()
            if system_message:
                self.rag_system.memory.chat_memory.messages.append(system_message)
            return "Conversation memory has been reset.", True
            
        # Stats command
        elif command_lower == "stats":
            # Capture printed output
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                VectorStoreManager.print_document_statistics(self.rag_system.vector_store)
            
            output = f.getvalue()
            if not output.strip():
                output = "No document statistics available."
                
            return output, True
            
        # Reindex command
        elif command_lower == "reindex":
            result = self.rag_system.reindex_documents()
            if result:
                return "Documents have been successfully reindexed.", True
            else:
                return "Failed to reindex documents. Check the log for details.", True
        
        # Unknown command
        else:
            return f"Unknown command: {command}. Type 'help' to see available commands.", True

#############################################################################
# 3. RETRIEVAL SYSTEM
#############################################################################

class RetrievalHandler:
    """Handles document retrieval using multiple strategies."""
    
    def __init__(self, vector_store, embeddings, memory, context_processor):
        """Initialize the retrieval handler."""
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.memory = memory
        self.context_processor = context_processor
        
        # Create retrievers for different strategies
        self.retrievers = self._create_retrievers()
    
    def _create_retrievers(self) -> Dict[RetrievalStrategy, Any]:
        """Create retrievers for different strategies."""
        retrievers = {}
        
        # Semantic search retriever
        retrievers[RetrievalStrategy.SEMANTIC] = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.RETRIEVER_K}
        )
        
        # MMR retriever for diversity
        retrievers[RetrievalStrategy.MMR] = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": Config.RETRIEVER_K,
                "fetch_k": Config.RETRIEVER_K * 3,
                "lambda_mult": 0.5
            }
        )
        
        # For keyword and hybrid search, we'll implement custom methods
        
        return retrievers
    
    def process_query(self, query_analysis: Dict[str, Any]) -> str:
        """
        Process a query using the appropriate retrieval strategy.
        
        Args:
            query_analysis: The analysis from InputProcessor
            
        Returns:
            Response string
        """
        query_type = query_analysis["query_type"]
        original_query = query_analysis["original_query"]
        expanded_queries = query_analysis.get("expanded_queries", [original_query])
        
        logger.info(f"Processing {query_type.value} query: {original_query}")
        
        # Select retrieval strategy based on query type
        retrieval_strategy = self._select_retrieval_strategy(query_type)
        logger.info(f"Selected retrieval strategy: {retrieval_strategy.value}")
        
        # Retrieve relevant documents
        context_docs = self._retrieve_documents(expanded_queries, retrieval_strategy)
        
        # If no documents found, try a fallback strategy
        if not context_docs and retrieval_strategy != RetrievalStrategy.HYBRID:
            logger.info("No documents found, trying hybrid fallback strategy")
            context_docs = self._retrieve_documents(expanded_queries, RetrievalStrategy.HYBRID)
        
        # Process the retrieved documents
        context = self.context_processor.process_context(context_docs, query_analysis)
        
        # Generate response using RAG system
        input_dict = {
            "question": original_query,
            "context": context,
            "chat_history": self.memory.chat_memory.messages
        }
        
        # Generate response through LLM
        response = RAGSystem.stream_ollama_response(self._create_prompt(input_dict), Config.LLM_MODEL_NAME)
        
        # Update memory
        self.memory.chat_memory.add_user_message(original_query)
        self.memory.chat_memory.add_ai_message(response)
        
        return response
    
    def _select_retrieval_strategy(self, query_type: QueryType) -> RetrievalStrategy:
        """
        Select the appropriate retrieval strategy based on query type.
        """
        if Config.RETRIEVER_SEARCH_TYPE != "auto":
            # Use the configured strategy if not set to auto
            return RetrievalStrategy(Config.RETRIEVER_SEARCH_TYPE)
        
        # Select strategy based on query type
        if query_type == QueryType.FACTUAL:
            return RetrievalStrategy.HYBRID  # Precise for factual questions
        
        elif query_type == QueryType.PROCEDURAL:
            return RetrievalStrategy.SEMANTIC  # Procedures need semantic understanding
        
        elif query_type == QueryType.COMPARATIVE:
            return RetrievalStrategy.MMR  # Diverse results for comparison
        
        elif query_type == QueryType.EXPLORATORY:
            return RetrievalStrategy.MMR  # Diverse results for exploration
        
        else:
            return RetrievalStrategy.HYBRID  # Default to hybrid
    
    def _retrieve_documents(self, queries: List[str], strategy: RetrievalStrategy) -> List[Document]:
        """
        Retrieve documents using the specified strategy.
        
        Args:
            queries: List of query strings (original and expanded)
            strategy: Retrieval strategy to use
            
        Returns:
            List of retrieved documents
        """
        all_docs = []
        seen_ids = set()  # Track document IDs to avoid duplicates
        
        # Process each query (original + expanded)
        for query in queries:
            try:
                if strategy == RetrievalStrategy.SEMANTIC or strategy == RetrievalStrategy.MMR:
                    # Use the appropriate retriever
                    docs = self.retrievers[strategy].invoke(query)
                    
                elif strategy == RetrievalStrategy.KEYWORD:
                    # Use keyword-based retrieval
                    docs = self._keyword_retrieval(query)
                    
                elif strategy == RetrievalStrategy.HYBRID:
                    # Use hybrid retrieval (combination of semantic and keyword)
                    docs = self._hybrid_retrieval(query)
                
                # Add unique documents to the result list
                for doc in docs:
                    # Create a unique ID based on content and source
                    doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
            
            except Exception as e:
                logger.error(f"Error retrieving documents for query '{query}': {e}")
        
        # Sort docs by relevance score if available
        all_docs = self._rerank_documents(all_docs, queries[0])
        
        logger.info(f"Retrieved {len(all_docs)} unique documents")
        return all_docs
    
    def _keyword_retrieval(self, query: str) -> List[Document]:
        """
        Perform keyword-based retrieval.
        
        Args:
            query: Query string
            
        Returns:
            List of documents
        """
        # Get all documents from the vector store
        all_docs = self.vector_store.get()
        
        if not all_docs or not all_docs.get('documents'):
            return []
            
        documents = all_docs.get('documents', [])
        metadatas = all_docs.get('metadatas', [])
        ids = all_docs.get('ids', [])
        
        # Extract keywords from the query
        tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Score documents based on keyword matches
        scored_docs = []
        for i, doc_text in enumerate(documents):
            if not doc_text:
                continue
                
            score = 0
            doc_lower = doc_text.lower()
            
            # Count exact keyword matches
            for keyword in keywords:
                # Count occurrences of the keyword
                count = doc_lower.count(keyword)
                score += count
                
                # Bonus for exact phrase match
                if query.lower() in doc_lower:
                    score += 10
            
            if score > 0:
                metadata = metadatas[i] if i < len(metadatas) else {}
                doc_id = ids[i] if i < len(ids) else str(i)
                
                # Create Document object
                doc = Document(
                    page_content=doc_text,
                    metadata={
                        **metadata,
                        'score': score,
                        'id': doc_id
                    }
                )
                scored_docs.append((score, doc))
        
        # Sort by score (descending) and return top k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Extract just the documents from the scored list
        result_docs = [doc for _, doc in scored_docs[:Config.RETRIEVER_K]]
        
        return result_docs
    
    def _hybrid_retrieval(self, query: str) -> List[Document]:
        """
        Perform hybrid retrieval (semantic + keyword).
        
        Args:
            query: Query string
            
        Returns:
            List of documents
        """
        # Get semantic search results
        semantic_docs = self.retrievers[RetrievalStrategy.SEMANTIC].invoke(query)
        
        # Get keyword search results
        keyword_docs = self._keyword_retrieval(query)
        
        # Combine results with weighting
        semantic_weight = 1 - Config.KEYWORD_RATIO
        keyword_weight = Config.KEYWORD_RATIO
        
        # Track documents by ID to avoid duplicates
        combined_docs = {}
        
        # Add semantic docs with their weights
        for i, doc in enumerate(semantic_docs):
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            
            # Inverse rank scoring (higher rank = lower score)
            semantic_score = semantic_weight * (1.0 - (i / len(semantic_docs)))
            
            # Store with score
            combined_docs[doc_id] = {
                "doc": doc,
                "score": semantic_score
            }
        
        # Add keyword docs with their weights
        for i, doc in enumerate(keyword_docs):
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            
            # Get original keyword score if available, otherwise use inverse rank
            keyword_score = (doc.metadata.get('score', 0) / max(1, max([d.metadata.get('score', 1) for d in keyword_docs])))
            
            # Adjust by weight
            keyword_score = keyword_weight * keyword_score
            
            # Update score if document already exists, otherwise add it
            if doc_id in combined_docs:
                combined_docs[doc_id]["score"] += keyword_score
            else:
                combined_docs[doc_id] = {
                    "doc": doc,
                    "score": keyword_score
                }
        
        # Sort by combined score
        sorted_docs = sorted(combined_docs.values(), key=lambda x: x["score"], reverse=True)
        
        # Return top K documents
        result_docs = [item["doc"] for item in sorted_docs[:Config.RETRIEVER_K]]
        
        return result_docs
    
    def _rerank_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            documents: List of documents to rerank
            query: Original query string
            
        Returns:
            Reranked document list
        """
        if not documents:
            return []
        
        # Embed the query
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            # Score each document
            scored_docs = []
            for doc in documents:
                # Try to get precomputed embedding if available
                doc_embedding = None
                if hasattr(doc, 'embedding') and doc.embedding is not None:
                    doc_embedding = doc.embedding
                else:
                    # Compute embedding
                    doc_embedding = self.embeddings.embed_query(doc.page_content)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                
                # Get keyword score if available
                keyword_score = doc.metadata.get('score', 0)
                
                # Combine scores (weighted average)
                combined_score = (similarity * 0.7) + (keyword_score * 0.3)
                
                scored_docs.append((combined_score, doc))
            
            # Sort by score (descending)
            scored_docs.sort(reverse=True, key=lambda x: x[0])
            
            # Return reranked documents
            return [doc for _, doc in scored_docs]
            
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            return documents  # Return original order if reranking fails
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)
    
    def _create_prompt(self, input_dict):
        """
        Create a prompt for the LLM based on query type.
        
        Args:
            input_dict: Dictionary with question, context, and chat history
            
        Returns:
            Formatted prompt string
        """
        template = """
        You are a helpful AI assistant answering questions about a collection of documents. 
        You have access to information contained in these documents, and you'll answer 
        questions about their content accurately and precisely.

        Guidelines for your responses:
        1. If you find the answer in the documents, respond directly and cite your sources with specific filenames
        2. If the documents contain partial information, use it and make clear where your information comes from
        3. If the question is about general knowledge unrelated to the documents, you can answer it like a normal conversation
        4. If you cannot find specific information in the documents, say "I don't have specific information about that in the documents" rather than making up information
        5. Always use specific information from the documents when available instead of giving generic answers
        6. If asked about a specific document or file by name, focus on information from that file and mention clearly whether that file is in your knowledge base
        7. Provide concise, direct answers that address the question's specific intent
        8. When citing information, mention the document name in a natural way, such as "According to [document name]..."

        Context from retrieved documents:
        {context}

        Chat History:
        {chat_history}

        Question: {question}

        Helpful Answer:
        """
        
        # Format chat history
        chat_history = ""
        for message in input_dict["chat_history"]:
            if isinstance(message, HumanMessage):
                chat_history += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                chat_history += f"AI: {message.content}\n"
        
        # Fill in the template
        prompt = template.format(
            context=input_dict["context"],
            chat_history=chat_history,
            question=input_dict["question"]
        )
        
        return prompt

#############################################################################
# 4. CONTEXT PROCESSING
#############################################################################

class ContextProcessor:
    """Processes retrieved documents into a coherent context for the LLM."""
    
    def __init__(self):
        """Initialize the context processor."""
        self.max_context_size = Config.MAX_CONTEXT_SIZE
        self.use_compression = Config.USE_CONTEXT_COMPRESSION
    
    def process_context(self, documents: List[Document], query_analysis: Dict[str, Any]) -> str:
        """
        Process retrieved documents into a coherent context.
        
        Args:
            documents: List of retrieved documents
            query_analysis: Analysis of the query
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        # Get query elements
        query_type = query_analysis["query_type"]
        keywords = query_analysis["keywords"]
        
        # Score and prioritize documents
        scored_docs = self._score_documents(documents, keywords, query_type)
        
        # Select documents to include based on priority and size constraints
        selected_docs = self._select_documents(scored_docs)
        
        # Format the selected documents
        formatted_context = self._format_documents(selected_docs)
        
        return formatted_context
    
    def _score_documents(self, documents: List[Document], keywords: List[str], query_type: QueryType) -> List[Tuple[Document, float, DocumentRelevance]]:
        """
        Score documents based on relevance to the query.
        
        Args:
            documents: List of documents
            keywords: List of query keywords
            query_type: Type of query
            
        Returns:
            List of tuples (document, score, relevance_level)
        """
        scored_docs = []
        
        for doc in documents:
            # Start with base score (if available in metadata)
            base_score = doc.metadata.get('score', 0.5)
            
            # Adjust score based on keyword matches
            keyword_score = 0
            content_lower = doc.page_content.lower()
            
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    # Count occurrences
                    count = content_lower.count(keyword.lower())
                    keyword_score += min(count / 10.0, 0.5)  # Cap at 0.5
            
            # Consider document length (prefer medium-sized chunks)
            length = len(doc.page_content)
            length_score = 0
            if 200 <= length <= 1000:
                length_score = 0.2  # Prefer medium chunks
            elif length > 1000:
                length_score = 0.1  # Long chunks are okay
            else:
                length_score = 0  # Short chunks less preferred
            
            # Adjust score based on query type and document content
            type_score = 0
            
            if query_type == QueryType.FACTUAL:
                # For factual queries, prefer documents with data, numbers, definitions
                if re.search(r'\b(?:defined?|mean|refer|is a|are a|definition)\b', content_lower):
                    type_score += 0.3
                if re.search(r'\d+', content_lower):
                    type_score += 0.2
                    
            elif query_type == QueryType.PROCEDURAL:
                # For procedural queries, prefer step-by-step content
                if re.search(r'\b(?:step|procedure|process|how to|guide|instruction)\b', content_lower):
                    type_score += 0.3
                if re.search(r'\b(?:first|second|third|next|then|finally)\b', content_lower):
                    type_score += 0.3
                    
            elif query_type == QueryType.CONCEPTUAL:
                # For conceptual queries, prefer explanations
                if re.search(r'\b(?:concept|theory|explanation|principle|understand|because)\b', content_lower):
                    type_score += 0.3
                    
            elif query_type == QueryType.COMPARATIVE:
                # For comparative queries, prefer content with comparisons
                if re.search(r'\b(?:compare|contrast|versus|vs|difference|similarity|advantage|disadvantage)\b', content_lower):
                    type_score += 0.4
            
            # Combine scores
            combined_score = (base_score * 0.4) + (keyword_score * 0.3) + (length_score * 0.1) + (type_score * 0.2)
            
            # Determine relevance level
            relevance = DocumentRelevance.MEDIUM  # Default
            if combined_score > 0.7:
                relevance = DocumentRelevance.HIGH
            elif combined_score < 0.3:
                relevance = DocumentRelevance.LOW
            
            scored_docs.append((doc, combined_score, relevance))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs
    
    def _select_documents(self, scored_docs: List[Tuple[Document, float, DocumentRelevance]]) -> List[Tuple[Document, DocumentRelevance]]:
        """
        Select documents to include in the context based on priority and size constraints.
        
        Args:
            scored_docs: List of scored documents
            
        Returns:
            List of selected documents with their relevance level
        """
        selected_docs = []
        current_size = 0
        
        # First, include HIGH relevance documents
        for doc, score, relevance in scored_docs:
            if relevance == DocumentRelevance.HIGH:
                doc_size = len(doc.page_content)
                
                # Skip if this single document is too large
                if doc_size > self.max_context_size:
                    # Try to compress if enabled
                    if self.use_compression:
                        compressed_content = self._compress_document(doc.page_content)
                        if len(compressed_content) <= self.max_context_size:
                            # Create a new document with compressed content
                            compressed_doc = Document(
                                page_content=compressed_content,
                                metadata=doc.metadata
                            )
                            selected_docs.append((compressed_doc, relevance))
                            current_size += len(compressed_content)
                    continue
                
                # Check if adding this document would exceed the limit
                if current_size + doc_size <= self.max_context_size:
                    selected_docs.append((doc, relevance))
                    current_size += doc_size
                else:
                    # Try compression if enabled
                    if self.use_compression:
                        compressed_content = self._compress_document(doc.page_content)
                        if current_size + len(compressed_content) <= self.max_context_size:
                            # Create a new document with compressed content
                            compressed_doc = Document(
                                page_content=compressed_content,
                                metadata=doc.metadata
                            )
                            selected_docs.append((compressed_doc, relevance))
                            current_size += len(compressed_content)
        
        # Then include MEDIUM relevance documents if space allows
        for doc, score, relevance in scored_docs:
            if relevance == DocumentRelevance.MEDIUM:
                doc_size = len(doc.page_content)
                
                # Check if adding this document would exceed the limit
                if current_size + doc_size <= self.max_context_size:
                    selected_docs.append((doc, relevance))
                    current_size += doc_size
                elif self.use_compression:
                    # Try compression
                    compressed_content = self._compress_document(doc.page_content)
                    if current_size + len(compressed_content) <= self.max_context_size:
                        # Create a new document with compressed content
                        compressed_doc = Document(
                            page_content=compressed_content,
                            metadata=doc.metadata
                        )
                        selected_docs.append((compressed_doc, relevance))
                        current_size += len(compressed_content)
        
        # Finally, include LOW relevance documents if there's still space
        if current_size < self.max_context_size * 0.8:  # Only if we have significant space left
            for doc, score, relevance in scored_docs:
                if relevance == DocumentRelevance.LOW:
                    doc_size = len(doc.page_content)
                    
                    # Check if adding this document would exceed the limit
                    if current_size + doc_size <= self.max_context_size:
                        selected_docs.append((doc, relevance))
                        current_size += doc_size
                    elif self.use_compression and current_size < self.max_context_size * 0.9:
                        # Try compression for low-relevance docs only if we have enough space
                        compressed_content = self._compress_document(doc.page_content)
                        if current_size + len(compressed_content) <= self.max_context_size:
                            # Create a new document with compressed content
                            compressed_doc = Document(
                                page_content=compressed_content,
                                metadata=doc.metadata
                            )
                            selected_docs.append((compressed_doc, relevance))
                            current_size += len(compressed_content)
        
        logger.info(f"Selected {len(selected_docs)} documents for context (size: {current_size}/{self.max_context_size})")
        return selected_docs
    
    def _compress_document(self, content: str) -> str:
        """
        Compress document content to fit in context window.
        
        Args:
            content: Document content to compress
            
        Returns:
            Compressed content
        """
        # Simple compression by removing extra whitespace
        compressed = re.sub(r'\s+', ' ', content).strip()
        
        # If still too long, try more aggressive compression
        if len(compressed) > self.max_context_size / 2:
            # Remove common filler phrases
            filler_phrases = [
                r'in order to',
                r'as a matter of fact',
                r'for the most part',
                r'for all intents and purposes',
                r'with regard to',
                r'in the event that',
                r'due to the fact that',
                r'in spite of the fact that',
                r'in the process of',
            ]
            
            for phrase in filler_phrases:
                # Replace with shorter alternatives
                if phrase == r'in order to':
                    compressed = re.sub(phrase, 'to', compressed)
                elif phrase == r'due to the fact that':
                    compressed = re.sub(phrase, 'because', compressed)
                elif phrase == r'in spite of the fact that':
                    compressed = re.sub(phrase, 'although', compressed)
                else:
                    # Remove other filler phrases
                    compressed = re.sub(phrase, '', compressed)
            
            # Replace repeated information if present
            # This is a simplistic approach; in a real system you might use
            # more sophisticated NLP to identify repetition
            sentences = re.split(r'(?<=[.!?])\s+', compressed)
            
            if len(sentences) > 5:
                # Check for similar sentences and remove duplicates
                unique_sentences = []
                sentence_fingerprints = set()
                
                for sentence in sentences:
                    # Create a simple fingerprint of the sentence
                    words = re.findall(r'\b\w+\b', sentence.lower())
                    if len(words) > 3:  # Only consider substantive sentences
                        fingerprint = ' '.join(sorted(words)[:5])  # Use first 5 sorted words as fingerprint
                        
                        # Keep sentence if fingerprint is unique
                        if fingerprint not in sentence_fingerprints:
                            sentence_fingerprints.add(fingerprint)
                            unique_sentences.append(sentence)
                    else:
                        # Always keep short sentences
                        unique_sentences.append(sentence)
                
                compressed = ' '.join(unique_sentences)
        
        # If still too long, truncate while preserving important parts
        if len(compressed) > self.max_context_size / 2:
            # Keep first and last parts as they're often most informative
            half_size = int(self.max_context_size / 4)
            compressed = f"{compressed[:half_size]} [...] {compressed[-half_size:]}"
        
        return compressed
    
    def _format_documents(self, selected_docs: List[Tuple[Document, DocumentRelevance]]) -> str:
        """
        Format the selected documents into a coherent context.
        
        Args:
            selected_docs: List of selected documents with relevance
            
        Returns:
            Formatted context string
        """
        if not selected_docs:
            return "No relevant documents found."
            
        formatted_docs = []
        unique_filenames = set()
        
        # Group documents by relevance
        high_docs = []
        medium_docs = []
        low_docs = []
        
        for doc, relevance in selected_docs:
            if relevance == DocumentRelevance.HIGH:
                high_docs.append(doc)
            elif relevance == DocumentRelevance.MEDIUM:
                medium_docs.append(doc)
            else:
                low_docs.append(doc)
            
            # Track unique filenames
            filename = doc.metadata.get('filename', os.path.basename(doc.metadata.get('source', 'Unknown')))
            unique_filenames.add(filename)
        
        # Add a summary of documents
        doc_count = len(selected_docs)
        file_count = len(unique_filenames)
        file_list = ', '.join(sorted(unique_filenames))
        
        summary = f"Retrieved {doc_count} relevant text segments from {file_count} document(s): {file_list}\n\n"
        formatted_docs.append(summary)
        
        # Add high relevance documents with clear formatting
        if high_docs:
            formatted_docs.append("--- HIGHLY RELEVANT INFORMATION ---")
            
            for i, doc in enumerate(high_docs):
                source = doc.metadata.get('source', 'Unknown source')
                filename = doc.metadata.get('filename', os.path.basename(source) if source != 'Unknown source' else 'Unknown file')
                
                # Include metadata if available
                page_info = f"page {doc.metadata.get('page', '')}" if doc.metadata.get('page', '') else ""
                
                metadata_line = f"Document {i+1} (from {filename} {page_info}):\n"
                formatted_text = f"{metadata_line}{doc.page_content}\n\n"
                formatted_docs.append(formatted_text)
        
        # Add medium relevance documents
        if medium_docs:
            formatted_docs.append("--- ADDITIONAL RELEVANT INFORMATION ---")
            
            for i, doc in enumerate(medium_docs):
                source = doc.metadata.get('source', 'Unknown source')
                filename = doc.metadata.get('filename', os.path.basename(source) if source != 'Unknown source' else 'Unknown file')
                
                # Include metadata if available
                page_info = f"page {doc.metadata.get('page', '')}" if doc.metadata.get('page', '') else ""
                
                metadata_line = f"Document {i+1+len(high_docs)} (from {filename} {page_info}):\n"
                formatted_text = f"{metadata_line}{doc.page_content}\n\n"
                formatted_docs.append(formatted_text)
        
        # Add low relevance documents
        if low_docs:
            formatted_docs.append("--- SUPPLEMENTARY INFORMATION ---")
            
            for i, doc in enumerate(low_docs):
                source = doc.metadata.get('source', 'Unknown source')
                filename = doc.metadata.get('filename', os.path.basename(source) if source != 'Unknown source' else 'Unknown file')
                
                # Include metadata if available
                page_info = f"page {doc.metadata.get('page', '')}" if doc.metadata.get('page', '') else ""
                
                metadata_line = f"Document {i+1+len(high_docs)+len(medium_docs)} (from {filename} {page_info}):\n"
                formatted_text = f"{metadata_line}{doc.page_content}\n\n"
                formatted_docs.append(formatted_text)
        
        return "\n".join(formatted_docs)

#############################################################################
# DOCUMENT PROCESSING
#############################################################################

class DocumentProcessor:
    """Handles loading, processing, and splitting documents."""
    
    @staticmethod
    def check_dependencies() -> bool:
        """Verify that required dependencies are installed."""
        missing_deps = []
        
        try:
            import docx2txt
        except ImportError:
            missing_deps.append("docx2txt (for DOCX files)")
        
        try:
            import pypdf
        except ImportError:
            missing_deps.append("pypdf (for PDF files)")
        
        try:
            import html2text
        except ImportError:
            missing_deps.append("html2text (for EPUB files)")
        
        try:
            import bs4
        except ImportError:
            missing_deps.append("beautifulsoup4 (for EPUB files)")
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
            logger.warning("Some document types may not load correctly.")
            logger.warning("Install missing dependencies with: pip install " + " ".join([d.split(' ')[0] for d in missing_deps]))
            return False
            
        return True
    
    @staticmethod
    def get_file_loader(file_path: str):
        """Returns appropriate loader based on file extension with error handling."""
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)  # Define filename variable

        try:
            if ext == '.pdf':
                return PyPDFLoader(file_path)
            elif ext in ['.docx', '.doc']:
                return Docx2txtLoader(file_path)
            elif ext in ['.ppt', '.pptx']:
                return UnstructuredPowerPointLoader(file_path)
            elif ext == '.epub':
                logger.info(f"Loading EPUB file: {filename}")
                try:
                    return UnstructuredEPubLoader(file_path)
                except Exception as e:
                    logger.warning(f"UnstructuredEPubLoader failed: {e}, trying alternative EPUB loader")
                    # Use DocumentProcessor instead of cls
                    docs = DocumentProcessor.load_epub(file_path)
                    if docs:
                        class CustomEpubLoader(BaseLoader):
                            def __init__(self, documents):
                                self.documents = documents
                            def load(self):
                                return self.documents
                        return CustomEpubLoader(docs)
                    return None
            elif ext in ['.txt', '.md', '.csv']:
                return TextLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {ext} for file {filename}")
                return None
        except Exception as e:
            logger.error(f"Error creating loader for {file_path}: {str(e)}")
            return None
    
    @classmethod
    def load_documents(cls, path: str, extensions: List[str] = None) -> List:
        """
        Load documents from specified path with specified extensions.
        Returns a list of documents or empty list if none found.
        """
        if extensions is None:
            extensions = Config.SUPPORTED_EXTENSIONS
            
        logger.info(f"Loading documents from: {path}")
        
        try:
            # Find all files with supported extensions
            all_files = []
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in extensions:
                        all_files.append(file_path)

            if not all_files:
                logger.warning(f"No compatible documents found in {path}")
                return []

            logger.info(f"Found {len(all_files)} compatible files")

            # Load each file with its appropriate loader
            all_documents = []
            for file_path in all_files:
                try:
                    logger.info(f"Loading: {os.path.basename(file_path)}")
                    loader = cls.get_file_loader(file_path)
                    if loader:
                        docs = loader.load()
                        if docs:
                            # Add source metadata to each document
                            for doc in docs:
                                if not hasattr(doc, 'metadata') or doc.metadata is None:
                                    doc.metadata = {}
                                doc.metadata['source'] = file_path
                                doc.metadata['filename'] = os.path.basename(file_path)
                                
                                # Add timestamp for sorting by recency if needed
                                try:
                                    doc.metadata['timestamp'] = os.path.getmtime(file_path)
                                except:
                                    doc.metadata['timestamp'] = 0

                            all_documents.extend(docs)
                            logger.info(f"Loaded {len(docs)} sections from {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    continue

            # Filter out empty documents
            valid_documents = [doc for doc in all_documents if doc.page_content and doc.page_content.strip()]
            
            if not valid_documents:
                logger.warning("No document content could be extracted successfully")
                
            logger.info(f"Successfully loaded {len(valid_documents)} total document sections")
            return valid_documents

        except Exception as e:
            logger.error(f"Document loading error: {e}")
            return []
    
    @staticmethod
    def load_epub(file_path: str):
        """
        Custom EPUB loader using ebooklib.
        Returns a list of LangChain Document objects.
        """
        try:
            from ebooklib import epub
            
            filename = os.path.basename(file_path)
            logger.info(f"Loading EPUB with custom loader: {filename}")
            
            # Load the EPUB file
            book = epub.read_epub(file_path)
            
            # Extract and process content
            documents = []
            h2t = html2text.HTML2Text()
            h2t.ignore_links = False
            
            # Get book title and metadata
            title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "Unknown Title"
            
            # Process each chapter/item
            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    # Extract HTML content
                    html_content = item.get_content().decode('utf-8')
                    
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Get plain text content
                    text = h2t.handle(str(soup))
                    
                    if text.strip():
                        # Create a document with metadata
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': file_path,
                                'filename': filename,
                                'title': title,
                                'chapter': item.get_name(),
                            }
                        )
                        documents.append(doc)
            
            logger.info(f"Extracted {len(documents)} sections from EPUB")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading EPUB file {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def split_documents(documents: List, chunk_size: int = None, chunk_overlap: int = None) -> List:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            
        Returns:
            List of document chunks
        """
        if not documents:
            logger.warning("No documents to split")
            return []
            
        if chunk_size is None:
            chunk_size = Config.CHUNK_SIZE
            
        if chunk_overlap is None:
            chunk_overlap = Config.CHUNK_OVERLAP
            
        logger.info(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
        
        try:
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            
            # Split documents
            chunked_documents = text_splitter.split_documents(documents)
            
            # Remove any empty chunks
            valid_chunks = [chunk for chunk in chunked_documents if chunk.page_content and chunk.page_content.strip()]
            
            # Log statistics
            logger.info(f"Created {len(valid_chunks)} chunks from {len(documents)} documents")
            
            return valid_chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return []

#############################################################################
# VECTOR STORE MANAGEMENT
#############################################################################

class VectorStoreManager:
    """Manages the vector database operations."""
    
    @staticmethod
    def get_embedding_device():
        """Determine the best available device for embeddings."""
        if torch.cuda.is_available():
            logger.info("Using CUDA GPU for embeddings")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using Apple Silicon MPS for embeddings")
            return 'mps'
        else:
            logger.info("Using CPU for embeddings")
            return 'cpu'
    
    @staticmethod
    def create_embeddings(model_name=None):
        """Create the embedding model."""
        if model_name is None:
            model_name = Config.EMBEDDING_MODEL_NAME
            
        device = VectorStoreManager.get_embedding_device()
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    @staticmethod
    def reset_chroma_db(persist_directory):
        """Reset the ChromaDB environment - handles file system operations safely."""
        logger.info(f"Resetting vector store at {persist_directory}")
        
        # Release resources via garbage collection
        import gc
        gc.collect()
        time.sleep(0.5)
        
        # Remove existing directory if it exists
        if os.path.exists(persist_directory):
            try:
                # Make directory writable first (for Windows compatibility)
                if sys.platform == 'win32':
                    for root, dirs, files in os.walk(persist_directory):
                        for dir in dirs:
                            os.chmod(os.path.join(root, dir), 0o777)
                        for file in files:
                            os.chmod(os.path.join(root, file), 0o777)
                
                # Try Python's built-in directory removal first
                shutil.rmtree(persist_directory)
            except Exception as e:
                logger.warning(f"Error removing directory with shutil: {e}")
                
                # Fallback to system commands
                try:
                    if sys.platform == 'win32':
                        os.system(f"rd /s /q \"{persist_directory}\"")
                    else:
                        os.system(f"rm -rf \"{persist_directory}\"")
                except Exception as e2:
                    logger.error(f"Failed to remove directory: {e2}")
                    return False
        
        # Create fresh directory structure
        try:
            os.makedirs(persist_directory, exist_ok=True)
            
            # Set appropriate permissions
            if sys.platform != 'win32':
                os.chmod(persist_directory, 0o755)
            
            # Create .chroma subdirectory for ChromaDB to recognize
            os.makedirs(os.path.join(persist_directory, ".chroma"), exist_ok=True)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create vector store directory: {e}")
            return False
    
    @classmethod
    def get_or_create_vector_store(cls, chunks=None, embeddings=None, persist_directory=None):
        """
        Get existing vector store or create a new one.
        
        Args:
            chunks: Document chunks to index (if creating new store)
            embeddings: Embedding model to use
            persist_directory: Directory to store the vector database
            
        Returns:
            The vector store or None if creation fails
        """
        if persist_directory is None:
            persist_directory = Config.PERSIST_PATH
            
        # If force reindex flag is set or directory doesn't exist, reset it
        if Config.FORCE_REINDEX or not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            cls.reset_chroma_db(persist_directory)
        
        # Try to load existing store if no chunks provided
        if chunks is None and os.path.exists(persist_directory) and os.listdir(persist_directory):
            try:
                logger.info(f"Loading existing vector store from: {persist_directory}")
                vector_store = Chroma(
                    persist_directory=persist_directory, 
                    embedding_function=embeddings
                )
                return vector_store
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                # If loading fails, reset and prepare for recreation
                cls.reset_chroma_db(persist_directory)
        
        # Create new vector store if chunks are provided
        if chunks:
            try:
                logger.info(f"Creating new vector store with {len(chunks)} chunks")
                
                # Configure ChromaDB properly
                chroma_settings = Settings(
                    anonymized_telemetry=False,
                    persist_directory=persist_directory
                )
                
                # Create the vector store with the chunks
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_directory,
                    collection_metadata={"hnsw:space": "cosine"},
                    client_settings=chroma_settings
                )
                
                # Explicitly persist to ensure it's saved
                if hasattr(vector_store, 'persist'):
                    vector_store.persist()
                    logger.info("Vector store persisted successfully")
                
                return vector_store
                
            except sqlite3.OperationalError as sqlerr:
                logger.error(f"Database error: {sqlerr}")
                return None
                
            except Exception as e:
                logger.error(f"Error creating vector store: {e}")
                return None
        
        logger.error("Cannot create or load vector store - no chunks provided and no existing store")
        return None
    
    @staticmethod
    def print_document_statistics(vector_store):
        """Print statistics about indexed documents."""
        if not vector_store:
            logger.warning("No vector store available for statistics")
            return
            
        try:
            all_docs = vector_store.get()
            if not all_docs or not all_docs.get('documents'):
                logger.warning("Vector store appears to be empty")
                return
                
            all_metadata = all_docs.get('metadatas', [])
            
            # Count documents by filename
            doc_counts = {}
            for metadata in all_metadata:
                if metadata and 'filename' in metadata:
                    filename = metadata['filename']
                    doc_counts[filename] = doc_counts.get(filename, 0) + 1
            
            total_chunks = len(all_docs.get('documents', []))
            unique_files = len(doc_counts)
            
            logger.info(f"Vector store contains {total_chunks} chunks from {unique_files} files")
            
            # Print file statistics
            print(f"\nKnowledge base contains {unique_files} documents ({total_chunks} total chunks):")
            for filename, count in sorted(doc_counts.items()):
                print(f"  - {filename}: {count} chunks")
                
            # Print only the most recently added document if timestamp is available
            recent_docs = []
            for i, metadata in enumerate(all_metadata):
                if metadata and 'timestamp' in metadata:
                    recent_docs.append((metadata['timestamp'], metadata.get('filename', 'Unknown')))
            
            if recent_docs:
                recent_docs.sort(reverse=True)
                # Get only the most recent document
                timestamp, filename = recent_docs[0]
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                print(f"\nMost recently added document: {filename} (added: {date_str}).")
                
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
    
    @staticmethod
    def verify_document_indexed(vector_store, doc_name):
        """Verify if a specific document is properly indexed."""
        if not vector_store:
            return False
            
        try:
            # Search for the document name in the vector store
            results = vector_store.similarity_search(f"information from {doc_name}", k=3)
            
            # Check if any results match this filename
            for doc in results:
                filename = doc.metadata.get('filename', '')
                if doc_name.lower() in filename.lower():
                    logger.info(f"Document '{doc_name}' is indexed in the vector store")
                    return True
                    
            logger.warning(f"Document '{doc_name}' was not found in the vector store")
            return False
                
        except Exception as e:
            logger.error(f"Error verifying document indexing: {e}")
            return False

#############################################################################
# 5. RESPONSE GENERATION
#############################################################################

class RAGSystem:
    """Manages the RAG processing pipeline."""
    
    @staticmethod
    def format_docs(docs):
        """Format retrieved documents for inclusion in the prompt."""
        if not docs:
            return "No relevant documents found."
            
        formatted_docs = []
        unique_filenames = set()
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown source')
            filename = doc.metadata.get('filename', os.path.basename(source) if source != 'Unknown source' else 'Unknown file')
            unique_filenames.add(filename)
            
            # Include metadata if available
            page_info = f"page {doc.metadata.get('page', '')}" if doc.metadata.get('page', '') else ""
            chunk_info = f"chunk {i+1}/{len(docs)}"
            
            metadata_line = f"Document {i+1} (from {filename} {page_info} {chunk_info}):\n"
            formatted_text = f"{metadata_line}{doc.page_content}\n\n"
            formatted_docs.append(formatted_text)

        # Add a summary of documents
        summary = f"Retrieved {len(docs)} chunks from {len(unique_filenames)} files: {', '.join(unique_filenames)}\n\n"
        
        return summary + "\n".join(formatted_docs)
    
    @staticmethod
    def stream_ollama_response(prompt, model_name=None, base_url=None):
        """Stream response from Ollama API with token-by-token output."""
        if model_name is None:
            model_name = Config.LLM_MODEL_NAME
            
        if base_url is None:
            base_url = Config.OLLAMA_BASE_URL
            
        url = f"{base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": True
        }

        full_response = ""

        try:
            # Test connection to Ollama API
            test_url = f"{base_url}/api/tags"
            try:
                test_response = requests.get(test_url, timeout=5)
                if test_response.status_code != 200:
                    logger.error(f"Ollama API unavailable: HTTP {test_response.status_code}")
                    return "Error: Could not connect to Ollama API. Make sure Ollama is running."
            except requests.RequestException as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                return "Error: Could not connect to Ollama API. Make sure Ollama is running and accessible."

            # Process the streaming response
            with requests.post(url, headers=headers, json=data, stream=True, timeout=30) as response:
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return f"Error: Failed to generate response (HTTP {response.status_code})"

                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line.decode('utf-8'))
                            if 'response' in json_line:
                                token = json_line['response']
                                full_response += token
                                # print(token, end='', flush=True) # Uncomment for real-time streaming

                            if json_line.get('done', False):
                                break
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON from Ollama API: {line}")
        except Exception as e:
            logger.error(f"Error during Ollama request: {e}")
            return f"Error: {str(e)}"
        # print () # Uncomment for real-time streaming newline
        return full_response

#############################################################################
# MAIN APPLICATION
#############################################################################

class CustomRAG:
    """Main RAG application class for command line interface."""
    
    def __init__(self):
        """Initialize the RAG application."""
        self.vector_store = None
        self.embeddings = None
        self.memory = None
        self.input_processor = None
        self.context_processor = None
        self.retrieval_handler = None
        self.conversation_handler = None
        self.command_handler = None
        self.query_router = None
    
    def initialize(self):
        """Set up all components of the RAG system."""
        # Setup configuration
        Config.setup()
        
        # Check dependencies
        DocumentProcessor.check_dependencies()
        
        # Initialize components
        self.input_processor = InputProcessor()
        self.context_processor = ContextProcessor()
        
        # Create memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Add a system message to memory
        system_message = SystemMessage(content="I am an AI assistant that helps with answering questions about documents. I can also engage in casual conversation.")
        self.memory.chat_memory.messages.append(system_message)
        
        # Create embeddings
        try:
            self.embeddings = VectorStoreManager.create_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return False
            
        # Load or create vector store
        if Config.FORCE_REINDEX or not os.path.exists(Config.PERSIST_PATH) or not os.listdir(Config.PERSIST_PATH):
            logger.info("Creating new vector store")
            
            # Load and process documents
            documents = DocumentProcessor.load_documents(Config.DATA_PATH)
            if not documents:
                logger.error("No documents found to index")
                return False
                
            chunks = DocumentProcessor.split_documents(documents)
            if not chunks:
                logger.error("Failed to create document chunks")
                return False
                
            # Create vector store with chunks
            self.vector_store = VectorStoreManager.get_or_create_vector_store(chunks, self.embeddings)
            
            if self.vector_store:
                # Print statistics
                VectorStoreManager.print_document_statistics(self.vector_store)
        else:
            # Load existing vector store
            logger.info("Loading existing vector store")
            self.vector_store = VectorStoreManager.get_or_create_vector_store(None, self.embeddings)
            
            if self.vector_store:
                # Print statistics
                VectorStoreManager.print_document_statistics(self.vector_store)
        
        if not self.vector_store:
            logger.error("Failed to initialize vector store")
            return False
        
        # Initialize handlers
        self.conversation_handler = ConversationHandler(self.memory)
        self.retrieval_handler = RetrievalHandler(self.vector_store, self.embeddings, self.memory, self.context_processor)
        self.command_handler = CommandHandler(self)
        
        # Initialize query router
        self.query_router = QueryRouter(
            self.conversation_handler,
            self.retrieval_handler,
            self.command_handler
        )
            
        return True
    
    def reindex_documents(self):
        """Reindex all documents in the data directory."""
        logger.info("Reindexing documents")
        print("Reindexing documents. This may take a while...")
        
        # Close existing resources
        if self.vector_store is not None:
            try:
                # Try to explicitly close Chroma client
                if hasattr(self.vector_store, '_collection') and self.vector_store._collection is not None:
                    if hasattr(self.vector_store._collection, '_client'):
                        if hasattr(self.vector_store._collection._client, 'close'):
                            self.vector_store._collection._client.close()
                
                self.vector_store = None
                
                # Force garbage collection
                import gc
                gc.collect()
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error closing vector store: {e}")
        
        # Reset the vector store directory
        if not VectorStoreManager.reset_chroma_db(Config.PERSIST_PATH):
            logger.error("Failed to reset vector store")
            return False
        
        # Load and process documents
        try:
            documents = DocumentProcessor.load_documents(Config.DATA_PATH)
            if not documents:
                logger.error("No documents found to index")
                return False
                
            chunks = DocumentProcessor.split_documents(documents)
            if not chunks:
                logger.error("Failed to create document chunks")
                return False
                
            # Create fresh embeddings
            self.embeddings = VectorStoreManager.create_embeddings()
                
            # Create vector store with chunks
            self.vector_store = VectorStoreManager.get_or_create_vector_store(chunks, self.embeddings)
            
            if not self.vector_store:
                logger.error("Failed to create vector store")
                return False
                
            # Print statistics
            VectorStoreManager.print_document_statistics(self.vector_store)
            
            # Reinitialize retrieval handler with new vector store
            self.retrieval_handler = RetrievalHandler(self.vector_store, self.embeddings, self.memory, self.context_processor)
            
            # Update query router with new retrieval handler
            self.query_router = QueryRouter(
                self.conversation_handler,
                self.retrieval_handler,
                self.command_handler
            )
                
            print(f"Reindexing completed successfully! Added {len(chunks)} chunks to the vector store.")
            return True
            
        except Exception as e:
            logger.error(f"Error during reindexing: {e}")
            return False
    
    def run_cli(self):
        """Run the interactive command line interface."""
        if not self.initialize():
            logger.error("Failed to initialize RAG system")
            return
            
        # Print banner and instructions
        print("\n" + "="*60)
        print("📚 CustomRAG - Advanced Document Assistant 📚")
        print("="*60)
        print("Ask questions about your documents using natural language.")
        print("Commands:")
        print("  - Type 'help' to see available commands")
        print("  - Type 'exit' or 'quit' to stop")
        print("  - Type 'clear' to reset the conversation memory")
        print("  - Type 'reindex' to reindex all documents")
        print("  - Type 'stats' to see document statistics")
        print("="*60 + "\n")
        
        # Main interaction loop
        while True:
            try:
                query = input("\nYour Question: ").strip()
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                # Process query
                start_time = time.time()
                
                print("\nThinking...\n")
                
                try:
                    # Process and route query
                    query_analysis = self.input_processor.analyze_query(query)
                    
                    # Route the query to appropriate handler
                    response, should_continue = self.query_router.route_query(query_analysis)
                    
                    if response:
                        print(f"Response: {response}")
                    
                    if not should_continue:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    print("\n[Error occurred during response generation]")
                
                end_time = time.time()
                logger.info(f"Query processed in {end_time - start_time:.2f} seconds")
                
            except KeyboardInterrupt:
                print("\nGoodbye! Have a great day!")
                break
            except Exception as e:
                logger.error(f"Error in CLI: {e}")
                print("\n[An unexpected error occurred. Please try again.]")

def main():
    """Entry point for the CustomRAG application."""
    try:
        app = CustomRAG()
        app.run_cli()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())