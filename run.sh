#!/bin/bash

# LLM-RAG Template - Interactive Document Query System
# This script demonstrates the core functionality of the RAG-LLM system

echo "🚀 LLM-RAG Template - Document Intelligence System"
echo "=================================================="
echo ""

# Step 1: Check dependencies
echo "📋 Checking dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required. Please install Python 3.10+"
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Please install from https://ollama.ai"
    echo "   Or modify config.py to use a different LLM provider"
    echo ""
    echo "To install Ollama on macOS:"
    echo "  brew install ollama"
    echo ""
    echo "For other systems, visit: https://ollama.ai/download"
    exit 1
fi

# Step 2: Start Ollama if needed
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚡ Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Step 3: Check for LLM model
echo "🤖 Checking for LLM model..."
if ! ollama list 2>/dev/null | grep -q "llama3.2:3b"; then
    echo "📦 Downloading Llama 3.2 model (this may take a few minutes)..."
    ollama pull llama3.2:3b
fi

# Step 4: Install Python dependencies if needed
echo "📚 Checking Python dependencies..."
if ! python3 -c "import chromadb" 2>/dev/null; then
    echo "📦 Installing required Python packages..."
    pip3 install chromadb sentence-transformers requests tqdm beautifulsoup4 PyPDF2 --quiet
fi

# Step 5: Check for documents
echo ""
echo "📂 Checking documents folder..."
DOC_COUNT=$(find documents -type f \( -name "*.pdf" -o -name "*.txt" -o -name "*.md" -o -name "*.html" \) 2>/dev/null | wc -l | tr -d ' ')

if [ "$DOC_COUNT" -eq 0 ]; then
    echo "⚠️  No documents found in ./documents/"
    echo ""
    echo "📝 Creating sample documents for demo..."

    # Create sample documents for demonstration
    cat > documents/rag_explained.txt << 'EOF'
RAG (Retrieval-Augmented Generation) combines the power of large language models
with external knowledge retrieval. This approach allows AI systems to access specific,
up-to-date information beyond their training data, significantly improving accuracy and relevance.

Key components of RAG:
1. Document Store: A collection of documents containing domain knowledge
2. Embedding Model: Converts text into numerical vectors for similarity search
3. Vector Database: Stores and indexes document embeddings for fast retrieval
4. Retrieval System: Finds relevant documents based on query similarity
5. Language Model: Generates responses using retrieved context

RAG is particularly useful for:
- Question answering over private documents
- Reducing hallucinations in LLM responses
- Keeping information up-to-date without retraining
- Domain-specific applications
EOF

    cat > documents/vector_databases.md << 'EOF'
# Vector Databases Overview

Vector databases enable semantic search by converting text into high-dimensional vectors.
They are essential for modern AI applications, particularly RAG systems.

## Popular Vector Databases

### ChromaDB
- **Pros**: Simple API, embedded mode, built-in persistence
- **Use Case**: Rapid prototyping, small to medium applications
- **Special Features**: Metadata filtering, automatic embeddings

### FAISS (Facebook AI Similarity Search)
- **Pros**: Very fast, CPU and GPU support, memory efficient
- **Use Case**: Large-scale applications, when speed is critical
- **Special Features**: Multiple index types, optimized algorithms

### Pinecone
- **Pros**: Fully managed, scales automatically, enterprise features
- **Use Case**: Production SaaS applications
- **Special Features**: Real-time updates, hybrid search

### Weaviate
- **Pros**: Full-featured, GraphQL API, module ecosystem
- **Use Case**: Complex applications, multi-modal search
- **Special Features**: Built-in ML models, data schemas

## Choosing the Right Database

Consider these factors:
1. **Scale**: How many vectors will you store?
2. **Speed**: What query latency is acceptable?
3. **Features**: Do you need metadata filtering? Hybrid search?
4. **Deployment**: Cloud, on-premise, or embedded?
5. **Cost**: Open source vs managed service
EOF

    cat > documents/llm_best_practices.txt << 'EOF'
Best Practices for LLM Applications

1. Prompt Engineering
   - Be specific and clear in instructions
   - Provide examples (few-shot learning)
   - Use structured formats for complex tasks
   - Test prompts with edge cases

2. Chunking Strategy
   - Optimal chunk size: 500-1000 tokens
   - Use overlapping chunks for context preservation
   - Consider semantic boundaries (paragraphs, sections)
   - Preserve metadata (source, page, timestamp)

3. Embedding Model Selection
   - Balance between quality and speed
   - Consider multilingual requirements
   - Test domain-specific models
   - Monitor embedding dimensions for storage

4. Context Management
   - Limit context to relevant information
   - Order matters: put important info first
   - Use summarization for long contexts
   - Implement context windowing

5. Response Validation
   - Implement fact-checking mechanisms
   - Add confidence scores
   - Use multiple models for consensus
   - Log and review edge cases

6. Performance Optimization
   - Cache frequent queries
   - Batch similar requests
   - Use streaming for long responses
   - Implement rate limiting

7. Security Considerations
   - Sanitize user inputs
   - Implement access controls
   - Audit LLM responses
   - Protect sensitive data in prompts

8. Monitoring and Evaluation
   - Track response quality metrics
   - Monitor latency and costs
   - Collect user feedback
   - A/B test different approaches
EOF

    echo "✅ Created 3 sample documents for demonstration"
    DOC_COUNT=3
fi

echo "📚 Found $DOC_COUNT document(s)"

# Step 6: Index documents
echo ""
echo "🔄 Building/Updating RAG index with ChromaDB..."
echo "   This creates vector embeddings of your documents for semantic search"
echo ""

# TODO: BUILD/CUSTOMIZE - This is where you customize the indexing process
python3 indexer.py --input documents --output chroma_db

if [ $? -ne 0 ]; then
    echo "❌ Error building index. Please check the error messages above."
    exit 1
fi

# Step 7: Interactive query loop
echo ""
echo "✅ System ready! You can now ask questions about your documents."
echo "   Type 'exit' to quit, 'help' for commands"
echo ""

# Function to show help
show_help() {
    echo ""
    echo "Available commands:"
    echo "  help     - Show this help message"
    echo "  index    - Rebuild the document index"
    echo "  stats    - Show document statistics"
    echo "  clear    - Clear the screen"
    echo "  exit     - Exit the program"
    echo ""
    echo "Or just type any question about your documents!"
    echo ""
}

# Function to show statistics
show_stats() {
    echo ""
    echo "📊 Document Statistics:"
    echo "   Total documents: $DOC_COUNT"
    if [ -d "chroma_db" ]; then
        echo "   Index size: $(du -sh chroma_db 2>/dev/null | cut -f1)"
    fi
    echo "   Supported formats: PDF, TXT, MD, HTML"
    echo ""
}

# Main query loop
while true; do
    echo -n "❓ What would you like to know about your documents? > "
    read -r user_query

    # Check for exit commands
    if [[ "$user_query" == "exit" ]] || [[ "$user_query" == "quit" ]]; then
        echo "👋 Goodbye!"
        break
    fi

    # Check for help
    if [[ "$user_query" == "help" ]]; then
        show_help
        continue
    fi

    # Check for index rebuild
    if [[ "$user_query" == "index" ]] || [[ "$user_query" == "reindex" ]]; then
        echo "🔄 Rebuilding index..."
        python3 indexer.py --input documents --output chroma_db
        if [ $? -eq 0 ]; then
            echo "✅ Index rebuilt successfully"
        else
            echo "❌ Error rebuilding index"
        fi
        continue
    fi

    # Check for stats
    if [[ "$user_query" == "stats" ]]; then
        show_stats
        continue
    fi

    # Check for clear
    if [[ "$user_query" == "clear" ]]; then
        clear
        echo "🚀 LLM-RAG Template - Document Intelligence System"
        echo "=================================================="
        echo ""
        continue
    fi

    # Skip empty queries
    if [[ -z "$user_query" ]]; then
        continue
    fi

    # Process the actual query
    echo ""
    echo "🔍 Searching documents for relevant information..."

    # TODO: BUILD/CUSTOMIZE - This is where you customize the query processing
    # The processor will:
    # 1. Search the vector database for relevant chunks
    # 2. Build context from top-k results
    # 3. Send to LLM with the query
    # 4. Return the answer

    # Call the main processor with the query
    response=$(python3 processor.py \
        --query "$user_query" \
        --db_path chroma_db \
        --collection documents \
        --model llama3.2:3b \
        --top_k 5 2>/dev/null)

    if [ $? -eq 0 ]; then
        # TODO: OUTPUT/CUSTOMIZE - This is where you customize the output format
        # You could:
        # - Save to a file
        # - Format as JSON
        # - Send to an API
        # - Generate a report

        echo ""
        echo "💡 Answer:"
        echo "----------"
        echo "$response"
        echo "----------"

        # Optional: Save query and response to output folder
        timestamp=$(date +%Y%m%d_%H%M%S)
        {
            echo "Query: $user_query"
            echo ""
            echo "Response:"
            echo "$response"
            echo ""
            echo "Timestamp: $(date)"
        } > "output/query_${timestamp}.txt"

        echo ""
        echo "📝 Response saved to output/query_${timestamp}.txt"
    else
        echo ""
        echo "❌ Error processing query. Please try again."
    fi

    echo ""
done

echo ""
echo "Thank you for using LLM-RAG Template!"