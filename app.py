# -*- coding: utf-8 -*-
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

from src.config import (
    PAGE_TITLE, 
    PAGE_ICON, 
    VECTORDB_DIR, 
    UPLOAD_DIR,
    TOP_K_RESULTS,
    GEMINI_API_KEY,
    LLM_TYPE,
    LOCAL_LLM_MODEL
)
from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.qa_engine import QAEngine
from src.helpful_functions import format_file_size, get_timestamp


# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS - Modern Dark Theme with High Contrast
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3561 0%, #1a1a2e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white !important;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Chat message containers - High Contrast */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(37, 99, 235, 0.25) 100%);
        border-left: 5px solid #3b82f6;
        color: #ffffff !important;
    }
    
    .user-message strong {
        color: #60a5fa !important;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.25) 0%, rgba(21, 128, 61, 0.25) 100%);
        border-left: 5px solid #22c55e;
        color: #ffffff !important;
    }
    
    .assistant-message strong {
        color: #4ade80 !important;
    }
    
    /* Ensure all text in chat messages is white */
    .chat-message * {
        color: #ffffff !important;
    }
    
    .chat-message small {
        color: #cbd5e1 !important;
    }
    
    /* Source box styling */
    .source-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.2) 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 0.5rem;
        border-left: 4px solid #f59e0b;
        font-size: 0.95rem;
        color: #fef3c7 !important;
    }
    
    .source-box * {
        color: #fef3c7 !important;
    }
    
    .source-box strong {
        color: #fbbf24 !important;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff !important;
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        padding: 0.75rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Slider styling */
    .stSlider>div>div>div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(102, 126, 234, 0.15);
        border-radius: 10px;
        color: #ffffff !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #60a5fa !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed rgba(102, 126, 234, 0.5);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: #ffffff !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
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
    
    # Header with modern design
    st.markdown('''
    <div class="main-header">📚 Smart Manual AI</div>
    <div style="text-align: center; color: #cbd5e1; font-size: 1.1rem; margin-bottom: 2rem;">
        <strong>Kullanım kılavuzlarınız artık akıllı! 🚀</strong><br>
        <span style="font-size: 0.95rem; color: #94a3b8;">PDF yükleyin, sorun ve anında cevap alın</span>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize models
    pdf_processor, embedding_model, vector_store, qa_engine = initialize_models()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; border-radius: 10px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%); margin-bottom: 1rem;">
            <h2 style="color: #a78bfa; margin: 0;">⚙️ Kontrol Paneli</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Database info
        doc_count = vector_store.get_document_count()
        st.metric("📊 Veritabanı", f"{doc_count} metin parçası")
        
        st.markdown("---")
        
        # PDF Upload Section
        st.markdown("### 📁 PDF Yükle")
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
        st.markdown("### 🎛️ AI Ayarları")
        
        # Show LLM type and status
        if LLM_TYPE == "gemini":
            if not GEMINI_API_KEY:
                st.error("⚠️ Gemini API key ayarlanmamış!")
                st.info("💡 .env dosyasına GEMINI_API_KEY ekleyerek yapay zeka destekli cevaplar alabilirsiniz.")
            else:
                st.success("✅ Gemini 2.0 Flash - Cloud LLM Aktif")
                st.caption("🌐 Veriler Google API'ye gönderiliyor")
        elif LLM_TYPE == "local":
            st.success(f"✅ Local LLM Aktif: {LOCAL_LLM_MODEL}")
            st.caption("🔒 Veriler cihazınızda kalıyor - Tamamen gizli")
            st.info("💡 Ollama kurulumu: [ollama.ai](https://ollama.ai)")
        else:
            st.warning(f"⚠️ Bilinmeyen LLM tipi: {LLM_TYPE}")
        
        top_k = st.slider(
            "📊 Cevap için kullanılacak kaynak sayısı",
            min_value=1,
            max_value=10,
            value=TOP_K_RESULTS,
            help="Sorunuza cevap verirken kaç farklı döküman parçasından bilgi toplanacağını belirler. Düşük değer (1-3) = odaklı cevap, Yüksek değer (5-10) = kapsamlı cevap"
        )
        
        st.markdown("---")
        
        # Database Management
        st.markdown("### 🗄️ Veritabanı")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Yenile", use_container_width=True):
                st.rerun()
        
        with col2:
            if doc_count > 0:
                if st.button("🗑️ Temizle", use_container_width=True):
                    vector_store.clear()
                    vector_store.save(str(VECTORDB_DIR))
                    st.session_state.chat_history = []
                    st.success("Temizlendi!")
                    st.rerun()
        
        # Info
        st.markdown("---")
        st.markdown("""
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #667eea;">
            <h4 style="color: #a78bfa; margin-top: 0;">💡 Nasıl Kullanılır?</h4>
            <ol style="color: #cbd5e1; line-height: 1.8;">
                <li><strong>PDF Yükle:</strong> Dosyalarınızı seçin</li>
                <li><strong>İşle:</strong> "Dosyaları İşle" butonuna tıklayın</li>
                <li><strong>Soru Sor:</strong> Sorunuzu yazın</li>
                <li><strong>Cevap Al:</strong> AI anında yardımcı olsun! &#x1F680;</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if doc_count == 0:
        # Welcome message
        st.info("Başlamak için lütfen sol menüden PDF dosyalarınızı yükleyin!")
        
        welcome_html = """
        <div style="padding: 2rem; border-radius: 15px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%); margin-top: 2rem;">
            <h2 style="color: #a78bfa;">Hoş Geldiniz! &#x1F44B;</h2>
            
            <p style="color: #cbd5e1; font-size: 1.1rem; line-height: 1.8;">
                <strong style="color: #60a5fa;">Smart Manual AI</strong> ile PDF kullanım talimatlarınız üzerinde akıllı soru-cevap yapabilirsiniz.
            </p>
            
            <div style="margin-top: 1.5rem;">
                <h3 style="color: #4ade80;">&#x2728; Özellikler:</h3>
                <ul style="color: #cbd5e1; line-height: 2;">
                    <li>&#x1F4C4; Çoklu PDF desteği</li>
                    <li>&#x1F5BC; Görsel ve diyagram analizi (OCR)</li>
                    <li>&#x1F1F9;&#x1F1F7; Türkçe + İngilizce dil desteği</li>
                    <li>&#x1F916; Gemini 2.0 Flash AI entegrasyonu</li>
                    <li>&#x1F50D; Akıllı anlamsal arama</li>
                    <li>&#x1F4DA; Kaynak referansları ve sayfa numaraları</li>
                    <li>&#x1F4AC; Sohbet geçmişi</li>
                </ul>
            </div>
            
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(34, 197, 94, 0.15); border-radius: 10px; border-left: 4px solid #22c55e;">
                <h4 style="color: #4ade80; margin: 0;">&#x1F680; Hızlı Başlangıç:</h4>
                <p style="color: #cbd5e1; margin-top: 0.5rem;">
                    Sol menüden PDF dosyalarınızı yükleyin ve hemen soru sormaya başlayın!
                </p>
            </div>
        </div>
        """
        st.markdown(welcome_html, unsafe_allow_html=True)
    
    else:
        # Q&A Interface
        qa_header_html = """
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h2 style="color: #a78bfa;">&#x1F4AC; Akıllı Asistan</h2>
            <p style="color: #94a3b8;">Sorunuzu yazın, anında cevap alın</p>
        </div>
        """
        st.markdown(qa_header_html, unsafe_allow_html=True)
        
        # Question input
        question = st.text_input(
            "Sorunuzu yazın:",
            placeholder="Örn: Hava sızdırmazlık parçaları V 4.4 için nelerdir?",
            key="question_input",
            label_visibility="collapsed"
        )
        
        # Ask button - centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            ask_button = st.button("&#x1F50D; Sor", type="primary", use_container_width=True)
        
        # Clear button - centered below
        if st.session_state.chat_history:
            col4, col5, col6 = st.columns([1, 2, 1])
            with col5:
                if st.button("&#x1F504; Temizle", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Process question
        if ask_button and question:
            with st.spinner("&#x1F914; Düşünüyorum..."):
                result = qa_engine.answer_question(
                    question,
                    k=top_k
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
            st.markdown("### &#x1F4DC; Sohbet Geçmişi", unsafe_allow_html=True)
            st.markdown("")
            
            # Show in reverse order (newest first)
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                # Question
                question_html = f"""
                <div class="chat-message user-message">
                    <strong style="color: #60a5fa; font-size: 1.1rem;">&#x1F64B; Soru</strong> 
                    <small style="color: #cbd5e1 !important;">({chat['timestamp']})</small><br><br>
                    <div style="color: #ffffff; font-size: 1rem; line-height: 1.6;">
                        {chat['question']}
                    </div>
                </div>
                """
                st.markdown(question_html, unsafe_allow_html=True)
                
                # Answer  
                answer_html = f"""
                <div class="chat-message assistant-message">
                    <strong style="color: #4ade80; font-size: 1.1rem;">&#x1F916; Cevap</strong><br><br>
                    <div style="color: #ffffff; font-size: 1rem; line-height: 1.8; white-space: pre-line;">
                        {chat['answer']}
                    </div>
                </div>
                """
                st.markdown(answer_html, unsafe_allow_html=True)
                
                # Sources
                if chat['sources']:
                    with st.expander("&#x1F4CE; Kaynaklar", expanded=False):
                        for j, source in enumerate(chat['sources'], 1):
                            page_info = f"Sayfa {source['page_number']}" if source.get('page_number') != 'N/A' else "Sayfa bilinmiyor"
                            content_type = source.get('content_type', 'text').upper()
                            icon = "&#x1F5BC;" if content_type == 'IMAGE' else "&#x1F4DD;"
                            
                            st.markdown(f"""
                            <div class="source-box">
                                <strong style="color: #fbbf24;">{j}. {icon} [{content_type}]</strong> 
                                <span style="color: #fef3c7;">{source['source']} - {page_info}</span><br>
                                <small style="color: #fde68a;">(Parça {source['chunk_id'] + 1}, Benzerlik Skoru: {source['score']:.2f})</small>
                            </div>
                            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
