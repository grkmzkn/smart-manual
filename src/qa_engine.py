from typing import List, Dict, Optional
import google.generativeai as genai
import requests
import json

# Import modules
from src.config import (
    GEMINI_API_KEY, 
    GEMINI_MODEL, 
    MAX_CONTEXT_LENGTH,
    LLM_TYPE,
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_MODEL,
    SIMILARITY_THRESHOLD
)
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class QAEngine:
    """Question-Answering engine using RAG (Retrieval Augmented Generation)."""
    
    def __init__(
        self, 
        embedding_model: EmbeddingModel, 
        vector_store: VectorStore,
        api_key: Optional[str] = None,
        llm_type: Optional[str] = None
    ):
        """
        Initialize QA engine.
        
        Args:
            embedding_model: Embedding model instance
            vector_store: Vector store instance
            api_key: Gemini API key (optional, for cloud LLM)
            llm_type: "gemini" or "local" (optional, defaults to config)
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_type = (llm_type or LLM_TYPE).lower()
        
        # Initialize based on LLM type
        if self.llm_type == "gemini":
            self.api_key = api_key or GEMINI_API_KEY
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(GEMINI_MODEL)
            else:
                self.model = None
        elif self.llm_type == "local":
            self.local_url = LOCAL_LLM_BASE_URL
            self.local_model = LOCAL_LLM_MODEL
            self.model = None  # Not used for local LLM
            # Check if Ollama is running
            self._check_ollama_connection()
        else:
            raise ValueError(f"Invalid LLM_TYPE: {self.llm_type}. Use 'gemini' or 'local'")
    
    def _check_ollama_connection(self):
        """Check if Ollama server is accessible."""
        try:
            response = requests.get(f"{self.local_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✅ Local LLM connected: {self.local_url}")
            else:
                print(f"⚠️ Ollama server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Warning: Cannot connect to Ollama at {self.local_url}")
            print(f"   Make sure Ollama is running: 'ollama serve'")
            print(f"   Error: {e}")
    
    def retrieve_context(self, question: str, k: int = 3) -> List[Dict]:
        """
        Retrieve relevant document chunks for the question.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata and scores (filtered by threshold)
        """
        # Convert question to embedding
        question_embedding = self.embedding_model.embed_text(question)
        
        # Search for similar documents
        results = self.vector_store.search(question_embedding, k=k)
        
        # Filter by similarity threshold - remove irrelevant results
        context_docs = [
            {
                'content': content,
                'score': score,
                'metadata': metadata
            }
            for content, score, metadata in results
            if score <= SIMILARITY_THRESHOLD  # Only include results below threshold
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
            return "Üzgünüm, yüklenen belgelerde bu soruyla ilgili bir bilgi bulamadım. Sorunuz belgelerin kapsamı dışında olabilir."
        
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
        
        # Prepare system prompt
        system_prompt = """Sen bir teknik kullanım kılavuzu asistanısın. Verilen belge içeriğine dayanarak kullanıcının sorusunu Türkçe olarak net ve anlaşılır bir şekilde cevapla.

ÖNEMLİ TALİMATLAR:
- Belgeler arasında [GÖRSEL İÇERİĞİ] veya OCR ile çıkarılmış metinler var - bunlara DİKKAT ET!
- Listeler, numaralandırmalar, parça isimleri, teknik detaylar varsa MUTLAKA belirt
- Görsellerde ve tablolarda yer alan bilgiler ÇOK ÖNEMLİ - bunları atlamadan kullan
- Soruya tam ve eksiksiz cevap ver

FORMATLAMA KURALLARI:
- Cevabı düzgün paragraflar halinde yaz
- Liste öğelerini madde işareti (-) veya numara ile göster
- Her maddeyi yeni satıra yaz
- Gereksiz satır sonları KULLANMA
- Paragraflar arası boşluk için çift satır sonu kullan"""
        
        # Generate answer based on LLM type
        if self.llm_type == "gemini":
            return self._generate_with_gemini(system_prompt, context_text, question)
        elif self.llm_type == "local":
            return self._generate_with_local_llm(system_prompt, context_text, question)
    
    def _generate_with_gemini(self, system_prompt: str, context_text: str, question: str) -> str:
        """Generate answer using Gemini API."""
        if not self.model:
            return "⚠️ Gemini API yapılandırılmamış. Lütfen .env dosyasına GEMINI_API_KEY ekleyin."
        
        try:
            prompt = f"""{system_prompt}

BELGELER:
{context_text}

KULLANICI SORUSU: {question}

CEVAP (sadece belgelerdeki bilgilere dayanarak, düzenli formatta):"""

            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Gemini LLM hatası: {str(e)}\n\nBulunan ilgili metin:\n{context_text[:500]}..."
    
    def _generate_with_local_llm(self, system_prompt: str, context_text: str, question: str) -> str:
        """Generate answer using local LLM (Ollama)."""
        try:
            print("🔄 Local LLM düşünüyor... (Bu işlem CPU'da 1-3 dakika sürebilir)")
            
            # Prepare prompt for local LLM
            prompt = f"""{system_prompt}

BELGELER:
{context_text}

KULLANICI SORUSU: {question}

CEVAP (sadece belgelerdeki bilgilere dayanarak, düzenli formatta):"""
            
            # Call Ollama API
            url = f"{self.local_url}/api/generate"
            payload = {
                "model": self.local_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused answers
                    "num_predict": 512,  # Max tokens (512 for faster response)
                    "num_ctx": 2048      # Context window
                }
            }
            
            # Increased timeout for slower CPUs (3 minutes)
            response = requests.post(url, json=payload, timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "Local LLM yanıt vermedi.")
                print("✅ Local LLM cevap verdi!")
                return answer
            else:
                return f"⚠️ Local LLM hatası (HTTP {response.status_code})\n\nBulunan ilgili metin:\n{context_text[:500]}..."
                
        except requests.exceptions.Timeout:
            return f"""⚠️ Local LLM zaman aşımı! Model çok yavaş yanıt veriyor.

💡 ÇÖZÜMLER:
1. Daha hızlı model kullanın:
   - ollama pull llama3.2:1b  (en hafif, en hızlı)
   - ollama pull phi3:mini     (hızlı ve kaliteli)
   
2. GPU'nuz varsa Ollama otomatik kullanmalı. Kontrol edin:
   - nvidia-smi  (NVIDIA GPU için)
   
3. Daha kısa sorular sorun veya TOP_K_RESULTS değerini düşürün

4. Veya Cloud LLM (Gemini) kullanın:
   - src/config.py → LLM_TYPE = "gemini"

Bulunan ilgili metin:
{context_text[:500]}..."""
                
        except requests.exceptions.ConnectionError:
            return f"⚠️ Local LLM'e bağlanılamadı. Ollama çalışıyor mu?\n\nKullanım: 'ollama serve' komutunu çalıştırın.\nModel: 'ollama pull {self.local_model}'\n\nBulunan ilgili metin:\n{context_text[:500]}..."
        except Exception as e:
            return f"Local LLM hatası: {str(e)}\n\nBulunan ilgili metin:\n{context_text[:500]}..."
    
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
