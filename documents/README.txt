=Ä Documents Folder
==================

Place your documents here for RAG indexing and analysis.

Supported formats:
- Text files (.txt)
- Markdown files (.md)
- PDF documents (.pdf)
- HTML files (.html)
- Word documents (.docx)
- JSON files (.json)

The system will automatically:
1. Scan this folder when you run ./run.sh
2. Parse and chunk your documents
3. Create vector embeddings
4. Build a searchable index

Tips:
- Organize documents in subfolders if needed
- Use descriptive filenames
- Remove unnecessary files to improve search quality
- Run 'index' command in the interface to rebuild after adding new documents

Sample documents will be created automatically if this folder is empty.