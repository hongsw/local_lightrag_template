#!/usr/bin/env python3
"""
Run script for the RAG API server.

Usage:
    python run.py              # Run the API server
    python run.py --index      # Index local files first, then run server
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="RAG API Server")
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index local PDF files before starting server"
    )
    parser.add_argument(
        "--path",
        default="./rag_raw_pdfs",
        help="Path to local files for indexing"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload"
    )

    args = parser.parse_args()

    if args.index:
        print(f"Indexing local files from: {args.path}")
        index_local_files(args.path)
        print("Indexing complete!")
        print()

    print(f"Starting RAG API server at http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print()

    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )


def index_local_files(path: str):
    """Index local files before starting the server."""
    from src.config import get_settings
    from src.connectors.local_files import LocalFilesConnector
    from src.processor import DocumentProcessor, ChunkConfig
    from src.rag_engine import RAGEngine

    settings = get_settings()

    # Initialize components
    connector = LocalFilesConnector(path)
    processor = DocumentProcessor(ChunkConfig(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ))
    engine = RAGEngine(
        working_dir=settings.lightrag_working_dir,
        openai_api_key=settings.openai_api_key,
        model_name=settings.openai_model,
        embedding_model=settings.embedding_model,
    )

    # Validate
    is_valid, error = connector.validate_config()
    if not is_valid:
        print(f"Error: {error}")
        sys.exit(1)

    # Load and process documents
    print("Loading documents...")
    documents = connector.load_documents()
    print(f"Found {len(documents)} documents")

    print("Processing documents...")
    processed = processor.process_documents(documents)
    print(f"Created {len(processed)} chunks")

    print("Indexing into LightRAG...")
    result = asyncio.run(engine.index_documents(processed))
    print(f"Indexed: {result['indexed']}, Errors: {result['errors']}")


if __name__ == "__main__":
    main()
