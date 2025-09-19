# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM-RAG (Large Language Model - Retrieval Augmented Generation) template that combines ChromaDB vector database with LLM integration for document analysis. The system indexes documents into embeddings, performs semantic search on queries, and uses an LLM to generate context-aware responses.

## Key Commands

### Running the System
```bash
# Main interactive interface (handles all setup automatically)
./run.sh

# Direct Python execution
python3 indexer.py --input documents --output chroma_db  # Build/rebuild vector index
python3 processor.py --query "your question" --db_path chroma_db --collection documents --model llama3.2:3b --top_k 5
```

### Dependencies
```bash
# Install all dependencies
pip3 install -r requirements.txt

# Core requirements: chromadb, sentence-transformers, requests, PyPDF2, beautifulsoup4
```

### Ollama Setup (for local LLM)
```bash
# macOS
brew install ollama
ollama serve  # Start service
ollama pull llama3.2:3b  # Download default model

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

## Architecture

### Core Components

1. **indexer.py**: DocumentIndexer class
   - Loads documents from `documents/` directory
   - Parses multiple formats (PDF, TXT, MD, HTML, DOCX, JSON)
   - Chunks text with configurable size/overlap
   - Creates embeddings using sentence-transformers
   - Stores in ChromaDB with metadata

2. **processor.py**: RAGProcessor class
   - Searches ChromaDB for relevant document chunks
   - Builds context from top-k results
   - Queries LLM (default: Ollama with llama3.2:3b)
   - Returns answer with source citations

3. **config.py**: Central configuration
   - Embedding model selection (default: all-MiniLM-L6-v2)
   - LLM provider/model configuration
   - Chunking parameters (size: 1000, overlap: 200)
   - RAG retrieval settings (top_k: 5)

### Data Flow

1. **Document Indexing Pipeline**:
   - Documents → Parse → Chunk → Embed → Store in ChromaDB

2. **Query Processing Pipeline**:
   - User Query → Embed → Search ChromaDB → Retrieve Top-K → Build Context → Query LLM → Return Response

### Key Extension Points

The codebase has `TODO: CUSTOMIZE` markers at critical customization points:

- **indexer.py:69**: Change embedding model for domain-specific needs
- **processor.py:68**: Add query preprocessing (expansion, rewriting)
- **processor.py:104**: Modify context formatting
- **processor.py:134**: Customize LLM prompt template
- **processor.py:148**: Add support for other LLM providers (OpenAI, Anthropic)
- **run.sh:182**: Customize indexing process
- **run.sh:283**: Customize query processing
- **run.sh:292**: Customize output format (JSON, API, reports)

## Testing

### Manual Testing
```bash
# Add test document
echo "Test content" > documents/test.txt

# Rebuild index
./run.sh
> index

# Query
> What does the test document contain?
```

### Verify Components
```bash
# Check Ollama
ollama list  # Should show llama3.2:3b

# Check ChromaDB index
ls -la chroma_db/  # Should contain index files after first run

# Check Python dependencies
python3 -c "import chromadb, sentence_transformers"
```

## Common Issues

- **Ollama connection refused**: Ensure `ollama serve` is running
- **Empty ChromaDB**: Documents folder needs supported file formats
- **Slow first run**: Model downloads happen on first use
- **Memory issues**: Reduce BATCH_SIZE in config.py or use smaller embedding model