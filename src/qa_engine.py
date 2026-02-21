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
from config.config import GEMINI_API_KEY, GEMINI_MODEL, USE_LLM
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
        context_docs: List[Dict],
        use_llm: bool = True
    ) -> str:
        """
        Generate answer based on question and retrieved context.
        
        Args:
            question: User question
            context_docs: Retrieved relevant documents
            use_llm: Whether to use LLM for answer generation
            
        Returns:
            Generated answer string
        """
        if not context_docs:
            return "Üzgünüm, yüklenen belgelerde bu soruyla ilgili bilgi bulamadım."
        
        # Combine context from documents
        context_text = "\n\n".join([doc['content'] for doc in context_docs])
        
        if use_llm and self.model:
            # Generate answer using Gemini
            try:
                prompt = f"""Sen bir kullanım kılavuzu asistanısın. Verilen belge içeriğine dayanarak kullanıcının sorusunu Türkçe olarak net ve anlaşılır bir şekilde cevapla.

Belgeler:
{context_text}

Kullanıcı Sorusu: {question}

Lütfen sadece verilen belgelerden yararlanarak soruyu cevapla. Eğer belgede cevap yoksa, bunu belirt."""

                response = self.model.generate_content(prompt)
                return response.text
                
            except Exception as e:
                return f"LLM hatası: {str(e)}\n\nBulunan ilgili metin:\n{context_text[:500]}..."
        else:
            # Simple context return (without LLM)
            return f"İlgili belgelerden bulunan bilgiler:\n\n{context_text}"
    
    def answer_question(
        self, 
        question: str, 
        k: int = 3,
        use_llm: bool = True
    ) -> Dict:
        """
        Complete QA pipeline: retrieve context and generate answer.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            use_llm: Whether to use LLM
            
        Returns:
            Dictionary with answer and source information
        """
        # Retrieve relevant context
        context_docs = self.retrieve_context(question, k=k)
        
        # Generate answer
        answer = self.generate_answer(question, context_docs, use_llm=use_llm)
        
        # Prepare response
        return {
            'answer': answer,
            'sources': [
                {
                    'source': doc['metadata']['source'],
                    'chunk_id': doc['metadata']['chunk_id'],
                    'score': doc['score']
                }
                for doc in context_docs
            ]
        }
