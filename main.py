# -*- coding: utf-8 -*-
"""
Command-line interface for Smart Manual application.
Terminal-based interface for testing PDF Q&A functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config import (
    VECTORDB_DIR, 
    UPLOAD_DIR, 
    TOP_K_RESULTS,
    LLM_TYPE,
    GEMINI_API_KEY,
    LOCAL_LLM_MODEL
)
from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.qa_engine import QAEngine
from utils.helpful_functions import validate_pdf_file, format_file_size


def initialize_system():
    """
    Initialize all system components.
    
    Returns:
        Tuple of (pdf_processor, embedding_model, vector_store, qa_engine)
    """
    print("🚀 Initializing Smart Manual system...")
    
    # Initialize components
    print("📦 Loading embedding model...")
    embedding_model = EmbeddingModel()
    
    print("🗄️  Initializing vector store...")
    vector_store = VectorStore()
    
    # Try to load existing database
    if (Path(VECTORDB_DIR) / "index.faiss").exists():
        print("📂 Loading existing vector database...")
        vector_store.load(str(VECTORDB_DIR))
        print(f"✅ Loaded {vector_store.get_document_count()} document chunks")
    
    # Initialize other components
    pdf_processor = PDFProcessor()
    qa_engine = QAEngine(embedding_model, vector_store)
    
    # Display LLM configuration
    print("\n🤖 LLM Configuration:")
    if LLM_TYPE == "gemini":
        if GEMINI_API_KEY:
            print("   ☁️  Cloud LLM: Gemini 2.5 Flash (Active)")
            print("   ⚠️  Data will be sent to Google API")
        else:
            print("   ❌ Cloud LLM: Gemini API key not configured")
            print("   💡 Add GEMINI_API_KEY to .env file")
    elif LLM_TYPE == "local":
        print(f"   🔒 Local LLM: {LOCAL_LLM_MODEL} (Private & Free)")
        print("   ✅ Data stays on your device")
        print("   💡 Make sure Ollama is running: 'ollama serve'")
    else:
        print(f"   ⚠️  Unknown LLM type: {LLM_TYPE}")
    
    print("\n✅ System initialized successfully!\n")
    
    return pdf_processor, embedding_model, vector_store, qa_engine


def process_pdf_file(pdf_path: str, pdf_processor, embedding_model, vector_store):
    """
    Process a PDF file and add to vector database.
    
    Args:
        pdf_path: Path to PDF file
        pdf_processor: PDFProcessor instance
        embedding_model: EmbeddingModel instance
        vector_store: VectorStore instance
    """
    print(f"\n📄 Processing PDF: {pdf_path}")
    
    # Validate PDF
    if not validate_pdf_file(pdf_path):
        print("❌ Invalid PDF file!")
        return
    
    # Get file size
    file_size = Path(pdf_path).stat().st_size
    print(f"📊 File size: {format_file_size(file_size)}")
    
    # Extract and process PDF
    print("📖 Extracting text from PDF...")
    documents = pdf_processor.process_pdf(pdf_path)
    print(f"✅ Created {len(documents)} text chunks")
    
    # Generate embeddings
    print("🔢 Generating embeddings...")
    texts = [doc['content'] for doc in documents]
    embeddings = embedding_model.embed_texts(texts, show_progress=True)
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    # Add to vector store
    print("💾 Adding to vector database...")
    vector_store.add_documents(embeddings, documents)
    
    # Save database
    print("💾 Saving vector database...")
    vector_store.save(str(VECTORDB_DIR))
    
    print(f"✅ PDF processed successfully!")
    print(f"📊 Total documents in database: {vector_store.get_document_count()}\n")


def ask_question(question: str, qa_engine):
    """
    Ask a question and get an answer.
    
    Args:
        question: Question text
        qa_engine: QAEngine instance
    """
    print(f"\n❓ Question: {question}")
    print("🔍 Searching for relevant information...\n")
    
    # Get answer
    result = qa_engine.answer_question(question, k=TOP_K_RESULTS)
    
    # Display answer
    print("=" * 80)
    print("🤖 ANSWER:")
    print("=" * 80)
    print(result['answer'])
    print("\n" + "=" * 80)
    
    # Display sources
    if result['sources']:
        print("\n📚 SOURCES:")
        for i, source in enumerate(result['sources'], 1):
            page_info = f"Page {source['page_number']}" if source.get('page_number') != 'N/A' else "Page N/A"
            content_type = source.get('content_type', 'text').upper()
            print(f"  {i}. [{content_type}] {source['source']} - {page_info} (Chunk {source['chunk_id'] + 1}, Score: {source['score']:.2f})")
    
    print("\n")


def main_menu():
    """Display main menu and handle user input."""
    print("\n" + "=" * 80)
    print("📚 SMART MANUAL - PDF Q&A SYSTEM")
    print("=" * 80)
    print("\nOptions:")
    print("  1. Process PDF file")
    print("  2. Ask a question")
    print("  3. Show database stats")
    print("  4. Clear database")
    print("  5. View image chunks (Debug)")
    print("  6. Exit")
    print("=" * 80)
    
    choice = input("\nEnter your choice (1-6): ").strip()
    return choice


def show_database_stats(vector_store):
    """Display vector database statistics."""
    print("\n📊 DATABASE STATISTICS:")
    print(f"  Total document chunks: {vector_store.get_document_count()}")
    print(f"  Embedding dimension: {vector_store.dimension}")
    print(f"  Database location: {VECTORDB_DIR}")
    
    print("\n🤖 LLM CONFIGURATION:")
    if LLM_TYPE == "gemini":
        status = "✅ Active" if GEMINI_API_KEY else "❌ Not configured"
        print(f"  Type: Cloud LLM (Gemini 2.0 Flash)")
        print(f"  Status: {status}")
        print(f"  Privacy: ⚠️  Data sent to Google API")
        print(f"  Cost: Pay per API call")
    elif LLM_TYPE == "local":
        print(f"  Type: Local LLM (Ollama)")
        print(f"  Model: {LOCAL_LLM_MODEL}")
        print(f"  Privacy: 🔒 100% Private (data stays local)")
        print(f"  Cost: ✅ Free")
    
    print("\n💡 To change LLM: Edit config/config.py")
    print("   - LLM_TYPE = \"gemini\"  (cloud, high quality)")
    print("   - LLM_TYPE = \"local\"   (private, free)")
    print()


def clear_database(vector_store):
    """Clear the vector database."""
    confirm = input("\n⚠️  Are you sure you want to clear the database? (yes/no): ").strip().lower()
    if confirm == 'yes':
        vector_store.clear()
        vector_store.save(str(VECTORDB_DIR))
        print("✅ Database cleared successfully!\n")
    else:
        print("❌ Operation cancelled.\n")


def view_image_chunks(vector_store):
    """View all image chunks in the database for debugging."""
    print("\n" + "=" * 80)
    print("🖼️  IMAGE CHUNKS IN DATABASE")
    print("=" * 80)
    
    if vector_store.get_document_count() == 0:
        print("\n⚠️  Database is empty!\n")
        return
    
    # Filter page number if needed
    page_filter = input("\nFilter by page number (press Enter for all pages): ").strip()
    
    image_chunks = []
    for i, (doc, metadata) in enumerate(zip(vector_store.documents, vector_store.metadata)):
        if metadata.get('content_type') == 'image':
            # Apply page filter if specified
            if page_filter and str(metadata.get('page_number')) != page_filter:
                continue
            
            image_chunks.append({
                'index': i,
                'content': doc,
                'metadata': metadata
            })
    
    if not image_chunks:
        if page_filter:
            print(f"\n⚠️  No image chunks found on page {page_filter}!\n")
        else:
            print("\n⚠️  No image chunks found in database!\n")
        return
    
    print(f"\nFound {len(image_chunks)} image chunk(s):\n")
    
    for i, chunk in enumerate(image_chunks, 1):
        meta = chunk['metadata']
        print("=" * 80)
        print(f"IMAGE CHUNK #{i}")
        print("-" * 80)
        print(f"Source: {meta.get('source', 'Unknown')}")
        print(f"Page: {meta.get('page_number', 'N/A')}")
        print(f"Chunk ID: {meta.get('chunk_id', 'N/A')}")
        print(f"Dimensions: {meta.get('dimensions', 'N/A')}")
        print(f"Format: {meta.get('format', 'N/A')}")
        print("-" * 80)
        print("EXTRACTED TEXT:")
        print(chunk['content'])
        print("=" * 80)
        print()
        
        # Ask if user wants to continue after each chunk
        if i < len(image_chunks):
            cont = input("Press Enter to see next image chunk, or 'q' to quit: ").strip().lower()
            if cont == 'q':
                break
    
    print(f"\n✅ Displayed {min(i, len(image_chunks))} of {len(image_chunks)} image chunks.\n")


def main():
    """Main CLI application loop."""
    # Initialize system
    pdf_processor, embedding_model, vector_store, qa_engine = initialize_system()
    
    # Main loop
    while True:
        choice = main_menu()
        
        if choice == '1':
            # Process PDF
            pdf_path = input("\nEnter PDF file path: ").strip()
            process_pdf_file(pdf_path, pdf_processor, embedding_model, vector_store)
            
        elif choice == '2':
            # Ask question
            if vector_store.get_document_count() == 0:
                print("\n⚠️  No documents in database. Please process a PDF first!\n")
                continue
            
            question = input("\nEnter your question: ").strip()
            if question:
                ask_question(question, qa_engine)
            
        elif choice == '3':
            # Show stats
            show_database_stats(vector_store)
            
        elif choice == '4':
            # Clear database
            clear_database(vector_store)
            
        elif choice == '5':
            # View image chunks
            view_image_chunks(vector_store)
            
        elif choice == '6':
            # Exit
            print("\n👋 Goodbye!\n")
            break
            
        else:
            print("\n❌ Invalid choice. Please try again.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
