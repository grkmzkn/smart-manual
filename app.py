"""
Streamlit web interface for Smart Manual application.
Modern web UI for PDF Q&A functionality.
"""

import streamlit as st
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config import (
    PAGE_TITLE, 
    PAGE_ICON, 
    VECTORDB_DIR, 
    UPLOAD_DIR,
    TOP_K_RESULTS,
    USE_LLM,
    GEMINI_API_KEY
)
from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.qa_engine import QAEngine
from utils.helpful_functions import format_file_size, get_timestamp


# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1E88E5;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_models():
    """
    Initialize and cache all models and components.
    
    Returns:
        Tuple of (pdf_processor, embedding_model, vector_store, qa_engine)
    """
    embedding_model = EmbeddingModel()
    vector_store = VectorStore()
    
    # Load existing database if available
    if (Path(VECTORDB_DIR) / "index.faiss").exists():
        vector_store.load(str(VECTORDB_DIR))
    
    pdf_processor = PDFProcessor()
    qa_engine = QAEngine(embedding_model, vector_store)
    
    return pdf_processor, embedding_model, vector_store, qa_engine


def process_uploaded_file(uploaded_file, pdf_processor, embedding_model, vector_store):
    """
    Process uploaded PDF file and add to vector database.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        pdf_processor: PDFProcessor instance
        embedding_model: EmbeddingModel instance
        vector_store: VectorStore instance
        
    Returns:
        Tuple of (success: bool, message: str, num_chunks: int)
    """
    try:
        # Save uploaded file
        file_path = Path(UPLOAD_DIR) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process PDF
        with st.spinner("📖 PDF işleniyor..."):
            documents = pdf_processor.process_pdf(str(file_path))
        
        # Generate embeddings
        with st.spinner("🔢 Metin vektörleri oluşturuluyor..."):
            texts = [doc['content'] for doc in documents]
            embeddings = embedding_model.embed_texts(texts, show_progress=False)
        
        # Add to vector store
        with st.spinner("💾 Veritabanına kaydediliyor..."):
            vector_store.add_documents(embeddings, documents)
            vector_store.save(str(VECTORDB_DIR))
        
        return True, "Başarılı", len(documents)
        
    except Exception as e:
        return False, str(e), 0


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">📚 Smart Manual Asistanı</div>', unsafe_allow_html=True)
    st.markdown("*PDF belgelerinizi yükleyin ve sorularınızı sorun!*")
    st.markdown("---")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize models
    pdf_processor, embedding_model, vector_store, qa_engine = initialize_models()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar ve Yönetim")
        
        # Database info
        doc_count = vector_store.get_document_count()
        st.metric("📊 Veritabanı", f"{doc_count} metin parçası")
        
        st.markdown("---")
        
        # PDF Upload Section
        st.subheader("📁 PDF Yükle")
        uploaded_files = st.file_uploader(
            "Kullanım talimatı PDF'lerini seçin",
            type=['pdf'],
            accept_multiple_files=True,
            help="Birden fazla PDF dosyası yükleyebilirsiniz"
        )
        
        if uploaded_files:
            if st.button("📤 Dosyaları İşle", type="primary"):
                success_count = 0
                total_chunks = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"İşleniyor: {file.name}")
                    
                    success, message, chunks = process_uploaded_file(
                        file, pdf_processor, embedding_model, vector_store
                    )
                    
                    if success:
                        success_count += 1
                        total_chunks += chunks
                        st.success(f"✅ {file.name} - {chunks} parça")
                    else:
                        st.error(f"❌ {file.name} - Hata: {message}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"🎉 {success_count}/{len(uploaded_files)} dosya işlendi!")
                st.info(f"Toplam {total_chunks} yeni parça eklendi")
                st.rerun()
        
        st.markdown("---")
        
        # Settings
        st.subheader("🎛️ Soru-Cevap Ayarları")
        
        use_llm = st.checkbox(
            "🤖 Gemini AI Kullan",
            value=USE_LLM,
            help="Gemini AI ile akıllı cevaplar oluştur",
            disabled=not bool(GEMINI_API_KEY)
        )
        
        if not GEMINI_API_KEY:
            st.warning("⚠️ Gemini API key ayarlanmamış. Temel mod aktif.")
            st.info("💡 .env dosyasına GEMINI_API_KEY ekleyin.")
        else:
            st.success("✅ Gemini AI hazır!")
        
        top_k = st.slider(
            "📊 Kullanılacak metin parçası",
            min_value=1,
            max_value=10,
            value=TOP_K_RESULTS,
            help="Cevap oluştururken kaç metin parçası kullanılsın"
        )
        
        st.markdown("---")
        
        # Database Management
        st.subheader("🗄️ Veritabanı Yönetimi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Yenile"):
                st.rerun()
        
        with col2:
            if doc_count > 0:
                if st.button("🗑️ Temizle"):
                    vector_store.clear()
                    vector_store.save(str(VECTORDB_DIR))
                    st.session_state.chat_history = []
                    st.success("Temizlendi!")
                    st.rerun()
        
        # Info
        st.markdown("---")
        st.markdown("""
        ### 💡 Nasıl Kullanılır?
        
        1. **PDF Yükle**: Yukarıdan PDF dosyalarınızı seçin
        2. **İşle**: "Dosyaları İşle" butonuna tıklayın
        3. **Soru Sor**: Sağ tarafta sorunuzu yazın
        4. **Cevap Al**: AI size yardımcı olsun!
        """)
    
    # Main content area
    if doc_count == 0:
        # Welcome message
        st.info("👈 Başlamak için lütfen sol menüden PDF dosyalarınızı yükleyin!")
        
        st.markdown("""
        ## Hoş Geldiniz! 👋
        
        **Smart Manual** ile PDF kullanım talimatlarınız üzerinde akıllı soru-cevap yapabilirsiniz.
        
        ### Özellikler:
        - 📄 Çoklu PDF desteği
        - 🇹🇷 Türkçe dil desteği
        - 🤖 Gemini AI entegrasyonu
        - 🔍 Akıllı metin arama
        - 📚 Kaynak referansları
        - 💬 Sohbet geçmişi
        
        ### Başlayın:
        Sol menüden PDF dosyalarınızı yükleyin ve hemen soru sormaya başlayın!
        """)
    
    else:
        # Q&A Interface
        st.subheader("💬 Soru Sorun")
        
        # Question input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_input(
                "Sorunuzu yazın:",
                placeholder="Örn: Bu cihaz nasıl açılır?",
                key="question_input",
                label_visibility="collapsed"
            )
        
        with col2:
            ask_button = st.button("🔍 Sor", type="primary", use_container_width=True)
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🔄 Sohbeti Temizle"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process question
        if ask_button and question:
            with st.spinner("🤔 Düşünüyorum..."):
                result = qa_engine.answer_question(
                    question,
                    k=top_k,
                    use_llm=use_llm and bool(GEMINI_API_KEY)
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'sources': result['sources'],
                    'timestamp': get_timestamp()
                })
                
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 📜 Sohbet Geçmişi")
            st.markdown("")
            
            # Show in reverse order (newest first)
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                # Question
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🙋 Soru</strong> <small style="color: #666;">({chat['timestamp']})</small><br>
                    {chat['question']}
                </div>
                """, unsafe_allow_html=True)
                
                # Answer
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Cevap</strong><br>
                    {chat['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Sources
                if chat['sources']:
                    with st.expander("📎 Kaynaklar", expanded=False):
                        for j, source in enumerate(chat['sources'], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>{j}.</strong> {source['source']} 
                                (Parça {source['chunk_id'] + 1}, Benzerlik Skoru: {source['score']:.2f})
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")


if __name__ == "__main__":
    main()
