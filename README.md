# 📚 Smart Manual - AI-Powered PDF Q&A System

Transform your PDF manuals into an intelligent question-answering assistant using RAG (Retrieval-Augmented Generation) technology.

## 🎯 Purpose

Smart Manual enables natural language queries on PDF documentation. Instead of manually searching through pages, simply ask questions and get instant answers with source references.

**Use Cases:**
- Technical manuals and user guides
- Industrial equipment documentation
- Vehicle service handbooks
- Medical device instructions
- Corporate policy documents

---

## ⚙️ Technology Stack

### LLM Options (Choose One)

**1. Google Gemini (Cloud)**
- High-quality responses
- Fast (1-3 seconds)
- Requires API key
- Best for: General use

**2. Ollama (Local)**
- 100% private and offline
- Free (no API costs)
- Slower on CPU
- Best for: Sensitive documents

### PDF Processing Methods

**Text Extraction:**
- `pdfplumber`: Extracts text and tables page-by-page
- Smart chunking: 1000 chars with 200 char overlap

**Image Processing (3 Options):**
1. **OCR (Tesseract)**: Free, offline text extraction from images
2. **Gemini Vision**: AI-powered image understanding (requires API)
3. **None**: Skip images (text-only, fastest)

### Vector Search

- **Embeddings**: Sentence Transformers (multilingual, 384-dim)
- **Database**: FAISS L2 similarity search
- **Retrieval**: Top-K most relevant chunks

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Configuration

Edit `src/config.py`:

```python
# Choose LLM type
LLM_TYPE = "local"  # or "gemini"

# For local LLM (requires Ollama)
LOCAL_LLM_MODEL = "qwen2.5:3b"  # or phi3:mini, llama3.2:1b

# Image processing mode
IMAGE_PROCESSING_MODE = "ocr"  # or "gemini" or "none"
```

### For Cloud LLM (Gemini)

Add to `.env`:
```
GEMINI_API_KEY=your-key-here
```
Get key: https://makersuite.google.com/app/apikey

### For Local LLM (Ollama)

```bash
# Install Ollama from https://ollama.ai/
# Download a model
ollama pull qwen2.5:3b
```

**Model Options:**
- `llama3.2:1b` - Fastest (~1GB)
- `phi3:mini` - Recommended (~2.3GB)
- `qwen2.5:3b` - Good quality (~2GB)

---

## 💻 Usage

### Web Interface

```bash
streamlit run app.py
```
Navigate to `http://localhost:8501`

### Command Line

```bash
python main.py
```

**Menu Options:**
1. Process PDF
2. Ask questions
3. View statistics
4. Clear database

---

## 📁 Project Structure

```
smart-manual/
├── src/
│   ├── config.py              # Settings
│   ├── pdf_processor.py       # PDF extraction
│   ├── image_processor.py     # OCR/Vision
│   ├── embeddings.py          # Vector embeddings
│   ├── vector_store.py        # FAISS database
│   ├── qa_engine.py           # RAG pipeline
│   └── helpful_functions.py   # Utilities
├── app.py                     # Web UI
├── main.py                    # CLI
└── requirements.txt
```

---

## 🔧 Configuration Options

| Setting | Default | Options |
|---------|---------|---------|
| LLM Type | `local` | `gemini`, `local` |
| Local Model | `qwen2.5:3b` | `phi3:mini`, `llama3.2:1b`, etc. |
| Image Mode | `ocr` | `gemini`, `ocr`, `none` |
| Chunk Size | 1000 | Any integer |
| Top-K Results | 3 | 1-10 recommended |

---

## 🙏 Credits

Built with:
- Google Gemini / Ollama (LLM)
- FAISS (Vector search)
- Sentence Transformers (Embeddings)
- pdfplumber (PDF parsing)
- Tesseract (OCR)
- Streamlit (Web UI)
