"""
LLM-RAG Template Configuration
TODO: CUSTOMIZE - Modify these settings for your use case
"""

# ChromaDB Settings
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and good quality
# Alternative models:
# - "all-mpnet-base-v2" - Better quality, slower
# - "all-distilroberta-v1" - Good balance
# - "instructor-base" - Domain-specific embeddings

# Document Processing
SUPPORTED_FORMATS = ['.txt', '.md', '.pdf', '.html', '.docx', '.json']
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
MAX_FILE_SIZE_MB = 100  # Maximum file size to process

# LLM Settings
LLM_PROVIDER = "ollama"  # Options: ollama, openai, anthropic, groq
LLM_MODEL = "llama3.2:3b"  # Model name
# Alternative models:
# - "mistral:7b" - Good alternative
# - "llama2:13b" - Larger, better quality
# - "codellama:7b" - For code-heavy content
# - "gpt-4" - OpenAI (requires API key)
# - "claude-3-sonnet" - Anthropic (requires API key)

LLM_TEMPERATURE = 0.7  # 0.0 = deterministic, 1.0 = creative
LLM_MAX_TOKENS = 2000  # Maximum response length
LLM_TIMEOUT = 60  # Seconds to wait for response

# RAG Settings
TOP_K_RESULTS = 5  # Number of chunks to retrieve
MIN_SIMILARITY_SCORE = 0.5  # Minimum similarity for retrieval

# API Keys (for cloud providers)
OPENAI_API_KEY = ""  # Set in environment variable
ANTHROPIC_API_KEY = ""  # Set in environment variable
GROQ_API_KEY = ""  # Set in environment variable

# Paths
DOCUMENTS_PATH = "./documents"
OUTPUT_PATH = "./output"
LOG_PATH = "./logs"
CACHE_PATH = "./cache"

# Performance
ENABLE_CACHING = True  # Cache LLM responses
BATCH_SIZE = 100  # For embedding generation
MAX_WORKERS = 4  # For parallel processing

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True
LOG_FILE = "llm_rag.log"