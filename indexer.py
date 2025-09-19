#!/usr/bin/env python3
"""
LLM-RAG Template - Document Indexer with ChromaDB
Builds vector database from documents for semantic search
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Error: ChromaDB not installed. Run: pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Optional progress bar

# Optional imports for document parsing
try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from bs4 import BeautifulSoup
    HAS_HTML = True
except ImportError:
    HAS_HTML = False


class DocumentIndexer:
    """Indexes documents into ChromaDB for RAG retrieval"""

    def __init__(self, input_dir: str, db_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the document indexer

        Args:
            input_dir: Directory containing documents to index
            db_path: Path to store ChromaDB persistent data
            embedding_model: Name of the sentence-transformers model to use
        """
        self.input_dir = Path(input_dir)
        self.db_path = db_path

        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        Path(db_path).mkdir(parents=True, exist_ok=True)

        print(f"🔧 Loading embedding model: {embedding_model}")
        print("   (First time may take a few minutes to download)")

        # Initialize embedding model
        # TODO: CUSTOMIZE - Change embedding model for your domain
        self.embedder = SentenceTransformer(embedding_model)

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Delete existing collection if it exists (for fresh indexing)
        try:
            self.client.delete_collection("documents")
        except:
            pass

        # Create new collection with embedding function
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"description": "Document embeddings for RAG"},
            embedding_function=None  # We'll add embeddings manually
        )

    def load_documents(self) -> List[Dict]:
        """
        Load all documents from the input directory

        Returns:
            List of document dictionaries with content and metadata
        """
        documents = []

        # TODO: CUSTOMIZE - Add more file types as needed
        patterns = ["*.txt", "*.md", "*.pdf", "*.html", "*.json"]

        print(f"📂 Scanning {self.input_dir} for documents...")

        for pattern in patterns:
            for file_path in self.input_dir.glob(f"**/{pattern}"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    doc = self.parse_document(file_path)
                    if doc and doc.get('content'):
                        documents.append(doc)
                        print(f"   ✓ Loaded: {file_path.name}")

        return documents

    def parse_document(self, file_path: Path) -> Dict:
        """
        Parse a single document based on its file type

        Args:
            file_path: Path to the document

        Returns:
            Dictionary with document content and metadata
        """
        # TODO: CUSTOMIZE - Add parsing logic for your document types

        try:
            if file_path.suffix in ['.txt', '.md']:
                # Text and Markdown files
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            elif file_path.suffix == '.pdf' and HAS_PDF:
                # PDF files
                content = ""
                try:
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            content += text + "\n"
                except Exception as e:
                    print(f"   ⚠️  Error reading PDF {file_path.name}: {e}")
                    content = None

            elif file_path.suffix == '.html' and HAS_HTML:
                # HTML files
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    content = soup.get_text()
                    # Clean up whitespace
                    lines = (line.strip() for line in content.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    content = ' '.join(chunk for chunk in chunks if chunk)

            elif file_path.suffix == '.json':
                # JSON files - extract text content
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert JSON to readable text
                    content = json.dumps(data, indent=2)

            else:
                content = None

            if content:
                return {
                    'path': str(file_path),
                    'name': file_path.name,
                    'content': content,
                    'type': file_path.suffix[1:] if file_path.suffix else 'unknown'
                }
            else:
                return None

        except Exception as e:
            print(f"   ⚠️  Error parsing {file_path}: {e}")
            return None

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval

        Args:
            text: The text to chunk
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        # TODO: CUSTOMIZE - Implement different chunking strategies
        # Options: by sentence, by paragraph, by token count, semantic chunking

        chunks = []

        # Handle small texts
        if len(text) <= chunk_size:
            return [text]

        # Create overlapping chunks
        start = 0
        while start < len(text):
            end = start + chunk_size

            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for separator in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_sep = text[start:end].rfind(separator)
                    if last_sep > chunk_size * 0.7:  # Found a good break point
                        end = start + last_sep + len(separator)
                        break

            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap if end < len(text) else end

        return chunks

    def create_chunks(self, documents: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
        """
        Create chunks from all documents

        Args:
            documents: List of document dictionaries

        Returns:
            Tuple of (chunks, metadata, ids)
        """
        all_chunks = []
        all_metadata = []
        all_ids = []

        for doc in documents:
            # TODO: CUSTOMIZE - Adjust chunk size and overlap for your use case
            chunks = self.chunk_text(doc['content'], chunk_size=1000, overlap=200)

            for i, chunk in enumerate(chunks):
                # Create unique ID for each chunk
                chunk_id = hashlib.md5(
                    f"{doc['path']}_{i}_{chunk[:50]}".encode()
                ).hexdigest()

                # Store metadata for each chunk
                metadata = {
                    'source': doc['name'],
                    'path': doc['path'],
                    'type': doc['type'],
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }

                all_chunks.append(chunk)
                all_metadata.append(metadata)
                all_ids.append(chunk_id)

        return all_chunks, all_metadata, all_ids

    def index_documents(self):
        """Main indexing pipeline"""

        # Load documents
        documents = self.load_documents()

        if not documents:
            print(f"❌ No documents found in {self.input_dir}")
            print("   Please add .txt, .md, .pdf, or .html files to the documents/ folder")
            return False

        print(f"✅ Loaded {len(documents)} document(s)")

        # Create chunks
        print("✂️  Creating document chunks...")
        chunks, metadata, ids = self.create_chunks(documents)
        print(f"   Created {len(chunks)} chunks")

        # Generate embeddings
        print("🧮 Generating embeddings (this may take a moment)...")
        embeddings = self.embedder.encode(chunks, show_progress_bar=True if tqdm else False)

        # Add to ChromaDB
        print("💾 Adding to ChromaDB vector database...")

        # Add in batches for better performance
        batch_size = 100
        iterator = range(0, len(chunks), batch_size)
        if tqdm:
            iterator = tqdm(iterator, desc="Indexing chunks")

        for i in iterator:
            batch_end = min(i + batch_size, len(chunks))
            self.collection.add(
                documents=chunks[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                metadatas=metadata[i:batch_end],
                ids=ids[i:batch_end]
            )

        print(f"✅ Successfully indexed {len(chunks)} chunks into ChromaDB")
        print(f"   Database location: {self.db_path}")
        print(f"   Collection: documents")
        print(f"   Ready for queries!")

        return True


def main():
    """Main entry point for command-line usage"""

    parser = argparse.ArgumentParser(
        description="Document Indexer - Build vector database from documents"
    )
    parser.add_argument("--input", default="documents", help="Input directory containing documents")
    parser.add_argument("--output", default="chroma_db", help="Output directory for ChromaDB")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")

    args = parser.parse_args()

    # Check if input directory exists
    if not Path(args.input).exists():
        print(f"❌ Input directory '{args.input}' does not exist")
        print(f"   Creating it now...")
        Path(args.input).mkdir(parents=True)
        print(f"   Please add documents to {args.input}/ and run again")
        sys.exit(1)

    # Initialize and run indexer
    indexer = DocumentIndexer(args.input, args.output, args.model)
    success = indexer.index_documents()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()