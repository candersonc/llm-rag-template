#!/usr/bin/env python3
"""
LLM-RAG Template - Main Query Processor with ChromaDB
This handles the RAG retrieval and LLM query processing
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Error: ChromaDB not installed. Run: pip install chromadb")
    sys.exit(1)

import requests


class RAGProcessor:
    """Main RAG processor that handles document retrieval and LLM queries"""

    def __init__(self, db_path: str, collection_name: str, model: str = "llama3.2:3b"):
        """
        Initialize the RAG processor

        Args:
            db_path: Path to ChromaDB persistent storage
            collection_name: Name of the document collection
            model: LLM model to use for generation
        """
        self.model = model
        self.db_path = db_path
        self.collection_name = collection_name

        # Initialize ChromaDB client with persistent storage
        try:
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )

            # Get the collection
            self.collection = self.client.get_collection(collection_name)

        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            print("Please run the indexer first to create the database.")
            sys.exit(1)

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search ChromaDB for relevant documents

        Args:
            query: The user's question
            n_results: Number of relevant chunks to retrieve

        Returns:
            List of relevant document chunks with metadata
        """
        # TODO: CUSTOMIZE - Add search preprocessing (e.g., query expansion)

        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            if results and results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0
                    })

            return formatted_results

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def build_context(self, results: List[Dict]) -> str:
        """
        Build context from search results

        Args:
            results: List of search results from ChromaDB

        Returns:
            Formatted context string for the LLM
        """
        # TODO: CUSTOMIZE - Modify context formatting for your use case

        if not results:
            return "No relevant documents found."

        context_parts = []
        for i, result in enumerate(results, 1):
            # Extract metadata
            source = result['metadata'].get('source', 'Unknown')
            chunk_index = result['metadata'].get('chunk_index', '')

            # Format each result
            context_parts.append(
                f"[Document {i}] Source: {source} (chunk {chunk_index})\n"
                f"{result['text']}\n"
            )

        return "\n---\n".join(context_parts)

    def query_llm(self, query: str, context: str) -> str:
        """
        Query the LLM with context from retrieved documents

        Args:
            query: The user's question
            context: Retrieved document context

        Returns:
            LLM's response
        """
        # TODO: CUSTOMIZE - Modify the prompt template for your use case
        prompt = f"""You are a helpful assistant that answers questions based on provided documents.
Use the following context from the documents to answer the question.
If the answer cannot be found in the context, say so clearly.

Context from documents:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above:"""

        try:
            # Call Ollama API
            # TODO: CUSTOMIZE - Add support for other LLM providers (OpenAI, Anthropic, etc.)
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2000,
                        "top_p": 0.9
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                return f"Error: LLM API returned status {response.status_code}"

        except requests.exceptions.Timeout:
            return "Error: LLM request timed out. The model might be loading or the response is taking too long."
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure Ollama is running (ollama serve)."
        except Exception as e:
            return f"Error calling LLM: {str(e)}"

    def process(self, query: str, n_results: int = 5) -> str:
        """
        Main processing pipeline: search, retrieve, generate

        Args:
            query: The user's question
            n_results: Number of documents to retrieve

        Returns:
            The final answer from the LLM
        """
        # Step 1: Search for relevant documents
        results = self.search(query, n_results)

        if not results:
            return ("No relevant documents found in the database. "
                   "Please make sure you have added documents to the documents/ folder "
                   "and run the indexer.")

        # Step 2: Build context from results
        context = self.build_context(results)

        # Add source information to response
        sources = list(set([r['metadata'].get('source', 'Unknown') for r in results]))
        sources_text = f"\n\n📚 Sources consulted: {', '.join(sources[:3])}"

        # Step 3: Query LLM with context
        answer = self.query_llm(query, context)

        # TODO: OUTPUT/CUSTOMIZE - Add any post-processing of the answer
        # For example: fact checking, formatting, adding citations

        return answer + sources_text


def main():
    """Main entry point for command-line usage"""

    parser = argparse.ArgumentParser(
        description="RAG Query Processor - Ask questions about your documents"
    )
    parser.add_argument("--query", required=True, help="Your question about the documents")
    parser.add_argument("--db_path", default="chroma_db", help="Path to ChromaDB storage")
    parser.add_argument("--collection", default="documents", help="Collection name in ChromaDB")
    parser.add_argument("--model", default="llama3.2:3b", help="LLM model to use")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")

    args = parser.parse_args()

    # Initialize processor
    processor = RAGProcessor(args.db_path, args.collection, args.model)

    # Process query
    answer = processor.process(args.query, args.top_k)

    # Output the answer (stdout for the shell script to capture)
    print(answer)


if __name__ == "__main__":
    main()