"""
Question-Answering engine module for Smart Manual application.
RAG-based answer generation using retrieved context and LLM.
"""

from typing import List, Dict, Optional
import google.generativeai as genai

# Import modules
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import GEMINI_API_KEY, GEMINI_MODEL, MAX_CONTEXT_LENGTH
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class QAEngine:
    """Question-Answering engine using RAG (Retrieval Augmented Generation)."""
    
    def __init__(
        self, 
        embedding_model: EmbeddingModel, 
        vector_store: VectorStore,
        api_key: Optional[str] = None
    ):
        """
        Initialize QA engine.
        
        Args:
            embedding_model: Embedding model instance
            vector_store: Vector store instance
            api_key: Gemini API key (optional)
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.api_key = api_key or GEMINI_API_KEY
        
        # Configure Gemini if API key is available
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(GEMINI_MODEL)
        else:
            self.model = None
    
    def retrieve_context(self, question: str, k: int = 3) -> List[Dict]:
        """
        Retrieve relevant document chunks for the question.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata and scores
        """
        # Convert question to embedding
        question_embedding = self.embedding_model.embed_text(question)
        
        # Search for similar documents
        results = self.vector_store.search(question_embedding, k=k)
        
        # Format results
        context_docs = [
            {
                'content': content,
                'score': score,
                'metadata': metadata
            }
            for content, score, metadata in results
        ]
        
        return context_docs
    
    def generate_answer(
        self, 
        question: str, 
        context_docs: List[Dict]
    ) -> str:
        """
        Generate answer based on question and retrieved context.
        
        Args:
            question: User question
            context_docs: Retrieved relevant documents
            
        Returns:
            Generated answer string
        """
        if not context_docs:
            return "Üzgünüm, yüklenen belgelerde bu soruyla ilgili bilgi bulamadım."
        
        # Combine context from documents with enhanced formatting
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            content_type = doc['metadata'].get('content_type', 'text').upper()
            page_num = doc['metadata'].get('page_number', 'N/A')
            
            # Format each chunk with metadata
            if content_type == 'IMAGE':
                header = f"--- KAYNAK {i}: [{content_type}] Sayfa {page_num} (GÖRSELDEN ÇIKARILMIŞ BİLGİ - ÖNEMLİ!) ---"
            else:
                header = f"--- KAYNAK {i}: [{content_type}] Sayfa {page_num} ---"
            
            context_parts.append(f"{header}\n{doc['content']}")
        
        context_text = "\n\n".join(context_parts)
        
        # Limit context length to control token usage
        if len(context_text) > MAX_CONTEXT_LENGTH:
            context_text = context_text[:MAX_CONTEXT_LENGTH] + "\n\n[Content truncated to limit token usage...]"
        
        # Generate answer using Gemini
        if not self.model:
            return "⚠️ Gemini API yapılandırılmamış. Lütfen .env dosyasına GEMINI_API_KEY ekleyin."
        
        try:
            prompt = f"""Sen bir teknik kullanım kılavuzu asistanısın. Verilen belge içeriğine dayanarak kullanıcının sorusunu Türkçe olarak net ve anlaşılır bir şekilde cevapla.

ÖNEMLİ TALİMATLAR:
- Belgeler arasında [GÖRSEL İÇERİĞİ] veya OCR ile çıkarılmış metinler var - bunlara DİKKAT ET!
- Listeler, numaralandırmalar, parça isimleri, teknik detaylar varsa MUTLAKA belirt
- Görsellerde ve tablolarda yer alan bilgiler ÇOK ÖNEMLİ - bunları atlamadan kullan
- Soruya tam ve eksiksiz cevap ver

BELGELER:
{context_text}

KULLANICI SORUSU: {question}

CEVAP (sadece belgelerdeki bilgilere dayanarak, eksiksiz ve detaylı):"""

            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"LLM hatası: {str(e)}\n\nBulunan ilgili metin:\n{context_text[:500]}..."
    
    def answer_question(
        self, 
        question: str, 
        k: int = 3
    ) -> Dict:
        """
        Complete QA pipeline: retrieve context and generate answer.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and source information
        """
        # Retrieve relevant context
        context_docs = self.retrieve_context(question, k=k)
        
        # Generate answer
        answer = self.generate_answer(question, context_docs)
        
        # Prepare response
        return {
            'answer': answer,
            'sources': [
                {
                    'source': doc['metadata']['source'],
                    'page_number': doc['metadata'].get('page_number', 'N/A'),
                    'chunk_id': doc['metadata']['chunk_id'],
                    'content_type': doc['metadata'].get('content_type', 'text'),
                    'score': doc['score']
                }
                for doc in context_docs
            ]
        }
