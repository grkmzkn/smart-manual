# 📚 Smart Manual - Intelligent PDF Q&A Assistant

> **An advanced RAG-based AI system that transforms your PDF user manuals into an intelligent question-answering assistant, supporting both cloud and privacy-focused local LLM options.**

## 🎯 Project Overview

Smart Manual is a **Production-Ready Retrieval-Augmented Generation (RAG) system** designed to extract knowledge from technical documentation, user manuals, and handbooks. Built with flexibility in mind, it offers both cloud-based AI (Google Gemini) and privacy-focused local LLM (Ollama) options, making it suitable for both public and confidential documents.

### Why Smart Manual?

Traditional PDF viewers offer only basic text search with exact keyword matching. Smart Manual revolutionizes document interaction by:

- **Understanding Context**: Uses semantic search to find relevant information even when your question uses different words than the manual
- **Multimodal Intelligence**: Extracts information from text, tables, diagrams, and images within PDFs
- **Privacy First**: Offers completely offline processing with local LLMs - your sensitive manuals never leave your machine
- **Cost Effective**: Choose between free local models or affordable cloud APIs based on your needs
- **Multi-language Support**: Optimized for Turkish and English with multilingual embedding models

### Use Cases

- 🏭 **Industrial Equipment Manuals**: Query maintenance procedures, troubleshooting steps, and technical specifications
- 🏥 **Medical Device Documentation**: Fast access to operation instructions and safety guidelines
- 🚗 **Vehicle Service Manuals**: Find repair procedures and part diagrams instantly
- 📱 **Consumer Electronics Guides**: Get setup help and feature explanations
- 🏢 **Corporate Policy Handbooks**: Navigate complex organizational documents with natural language

---

## ✨ Key Features

### 🤖 Dual LLM Architecture

**Cloud LLM (Google Gemini 2.5 Flash)**
- ☁️ State-of-the-art performance
- ⚡ Fast response times (1-3 seconds)
- 🎨 Advanced vision capabilities for images
- 💰 Pay-per-use pricing ($0.075/1M input tokens)

**Local LLM (Ollama)**
- 🔒 100% privacy - data never leaves your machine
- 💾 Runs completely offline
- 🆓 Zero API costs
- 🎛️ Model selection: from 1B to 7B+ parameters
- 🖥️ Optimized for CPU execution (GPU auto-detected)

### 📄 Advanced Multimodal PDF Processing

**Text Extraction**
- 📝 Page-by-page extraction with `pdfplumber`
- 🧩 Smart chunking with configurable overlap (1000 chars, 200 overlap)
- 📊 Automatic table detection and structured extraction
- 🔢 Page number tracking for precise source attribution

**Image Intelligence**
- 🎨 **Gemini Vision Mode**: Leverages Google's multimodal AI to understand diagrams, charts, and technical illustrations
- 🔍 **OCR Mode**: Tesseract-based text extraction from images with advanced preprocessing:
  - Grayscale conversion for better contrast
  - Image upscaling to 1200px for clarity
  - Contrast enhancement (2.0x) and sharpness boost (1.5x)
  - Smart line merging to fix fragmented sentences
  - Supports Turkish + English (configurable languages)
- 📄 **Page Context Enrichment**: Automatically adds surrounding text (300 chars) to image chunks for better context

**Processing Modes**
- `ocr`: Free Tesseract OCR (best for text-heavy diagrams)
- `gemini`: AI Vision API (best for complex visual content)
- `none`: Skip images (fastest, text-only)

### 🔍 Intelligent Retrieval System

**Vector Database (FAISS)**
- 🚀 Lightning-fast similarity search with L2 distance metric
- 💾 Persistent storage with `.faiss` index and `.pkl` metadata
- 🔄 Incremental updates without full reprocessing
- 📈 Scalable to thousands of document chunks

**Semantic Embeddings**
- 🌍 Multilingual model: `paraphrase-multilingual-MiniLM-L12-v2`
- 📐 384-dimensional dense vectors
- 🇹🇷 Optimized for Turkish language understanding
- 🎯 Top-K retrieval (default: 3 most relevant chunks)

**Context Management**
- 📏 Smart token limiting (8000 chars max to LLM)
- 📚 Source attribution with chunk IDs and page numbers
- 📊 Similarity scores for transparency (lower = better match)

### 🎨 Modern User Interfaces

**Streamlit Web UI**
- 🌑 Beautiful dark theme with gradient design
- 💬 Chat-style conversation interface
- 📤 Drag-and-drop PDF upload
- 📊 Real-time database statistics
- 🔧 Configurable retrieval settings (Top-K, context length)
- ✨ HTML-safe emoji rendering
- 📱 Responsive layout

**Command-Line Interface (CLI)**
- ⚡ Fast terminal-based interaction
- 📋 Interactive menu system
- 📊 Database statistics and health checks
- 🗑️ Database management (clear/rebuild)
- 🔍 Direct question input

---

## 🏗️ Architecture & Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM (Cloud)** | Google Gemini 2.5 Flash | Answer generation with vision |
| **LLM (Local)** | Ollama (qwen2.5:3b, phi3:mini, etc.) | Privacy-focused offline AI |
| **Embeddings** | Sentence Transformers (MiniLM-L12-v2) | Multilingual semantic vectors |
| **Vector DB** | FAISS IndexFlatL2 | Fast similarity search |
| **PDF Parsing** | pdfplumber | Text & table extraction |
| **Image Extraction** | PyMuPDF (fitz) | Extract images from PDFs |
| **OCR Engine** | Tesseract 5.0+ | Text recognition from images |
| **Image Processing** | Pillow (PIL) | Preprocessing & enhancement |
| **Web UI** | Streamlit 1.32.0+ | Interactive web interface |
| **Environment** | Python 3.8+ | Core runtime |

### Project Structure

```
smart-manual/
├── config/
│   └── config.py              # Centralized configuration
├── src/
│   ├── pdf_processor.py       # PDF text/table extraction
│   ├── image_processor.py     # OCR & Gemini Vision processing
│   ├── embeddings.py          # Sentence Transformer wrapper
│   ├── vector_store.py        # FAISS vector database
│   └── qa_engine.py           # RAG pipeline & LLM integration
├── utils/
│   └── helpful_functions.py   # Text cleaning, validation
├── data/
│   ├── uploaded/              # User-uploaded PDFs
│   └── vectordb/              # Persisted FAISS index
│       ├── index.faiss        # Vector index (embeddings)
│       └── documents.pkl      # Document chunks & metadata
├── app.py                     # Streamlit web interface
├── main.py                    # CLI interface
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
├── README.md                  # This file
└── SETUP_GUIDE.md             # Detailed installation guide

```

### RAG Pipeline Flow

```
┌─────────────────┐
│  PDF Document   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  PDF Processor (pdfplumber)         │
│  • Extract text by pages            │
│  • Detect and extract tables        │
│  • Extract images (PyMuPDF)         │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Image Processor (Optional)         │
│  • OCR: Tesseract + Preprocessing   │
│  • Vision: Gemini multimodal API    │
│  • Add page context (300 chars)     │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Chunking & Embedding               │
│  • Split into 1000-char chunks      │
│  • 200-char overlap for continuity  │
│  • Generate 384-dim embeddings      │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  FAISS Vector Store                 │
│  • Store embeddings + metadata      │
│  • Save to disk (index.faiss)       │
└─────────────────────────────────────┘

         [QUERY TIME]

┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Embed Question → Search FAISS      │
│  • Return Top-K similar chunks      │
│  • Include page numbers & scores    │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Context Assembly                   │
│  • Concatenate retrieved chunks     │
│  • Limit to 8000 chars              │
│  • Format with source attribution   │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  LLM Generation                     │
│  • Gemini: Cloud API                │
│  • Ollama: Local HTTP API           │
│  • Timeout: 180s (local), instant   │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Answer   │
└─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **Tesseract OCR** (optional, for image text extraction)
  - Windows: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt install tesseract-ocr tesseract-ocr-tur`
  - macOS: `brew install tesseract tesseract-lang`
- **Ollama** (optional, for local LLM)
  - [Download from ollama.ai](https://ollama.ai/)

### Installation

1. **Clone the repository**
   ```powershell
   git clone https://github.com/yourusername/smart-manual.git
   cd smart-manual
   ```

2. **Install Python dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```powershell
   # Create .env file from template
   cp .env.example .env
   ```

4. **Choose your LLM mode**

   **Option A: Cloud LLM (Gemini)**
   ```bash
   # Edit .env file
   GEMINI_API_KEY=your-api-key-here  # Get from https://makersuite.google.com/app/apikey

   # Edit config/config.py
   LLM_TYPE = "gemini"
   ```

   **Option B: Local LLM (Ollama)**
   ```powershell
   # Install Ollama from https://ollama.ai/
   
   # Download a model (choose based on your hardware)
   ollama pull phi3:mini        # Recommended for CPU (3.8B, ~2.3GB)
   ollama pull llama3.2:1b      # Fastest (1B, ~1GB)
   ollama pull qwen2.5:3b       # Balanced (3B, ~2GB)
   
   # Edit config/config.py
   LLM_TYPE = "local"
   LOCAL_LLM_MODEL = "phi3:mini"
   ```

### Usage

**Web Interface (Recommended)**
```powershell
streamlit run app.py
```
- Navigate to `http://localhost:8501`
- Upload your PDF in the sidebar
- Wait for processing (first time: 30-60s, cached: 3-5s)
- Start asking questions!

**Command Line Interface**
```powershell
python main.py
```
- Choose from interactive menu
- Option 1: Upload and process PDF
- Option 2: Ask questions
- Option 3: View database statistics
- Option 4: Clear database

---

## ⚙️ Configuration Guide

### Core Settings (`config/config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_TYPE` | `"local"` | LLM mode: `"gemini"` or `"local"` |
| `LOCAL_LLM_MODEL` | `"qwen2.5:3b"` | Ollama model name |
| `IMAGE_PROCESSING_MODE` | `"ocr"` | Image mode: `"ocr"`, `"gemini"`, or `"none"` |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | `3` | Number of chunks to retrieve |
| `MAX_CONTEXT_LENGTH` | `8000` | Max characters sent to LLM |
| `EMBEDDING_MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence Transformer model |

### LLM Model Recommendations

**For CPU (Ollama Local Models)**

| Model | Parameters | Size | Speed | Quality | Best For |
|-------|-----------|------|-------|---------|----------|
| `llama3.2:1b` | 1B | ~1GB | ⚡⚡⚡ | ⭐⭐ | Fastest possible |
| `phi3:mini` | 3.8B | ~2.3GB | ⚡⚡ | ⭐⭐⭐⭐ | **Recommended** |
| `llama3.2:3b` | 3B | ~2GB | ⚡⚡ | ⭐⭐⭐ | Balanced |
| `qwen2.5:3b` | 3B | ~2GB | ⚡ | ⭐⭐⭐⭐ | Good quality |
| `mistral:7b` | 7B | ~4GB | 🐌 (GPU only) | ⭐⭐⭐⭐⭐ | Best quality |

**Installation:**
```bash
ollama pull phi3:mini
```

### Environment Variables (`.env`)

Only **secrets** go here:
```bash
# Google Gemini API Key (only needed for cloud LLM)
GEMINI_API_KEY=your-actual-api-key-here
```

Get your key: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

---

## 📊 Performance Optimization

### Speed Optimization

**1. For Faster Processing:**
- Use `IMAGE_PROCESSING_MODE = "none"` if PDFs are text-only
- Reduce `TOP_K_RESULTS` to 2 for simpler queries
- Use smaller embedding model (requires code changes)

**2. For Better Quality:**
- Use `LLM_TYPE = "gemini"` for best answers
- Increase `TOP_K_RESULTS` to 5 for complex questions
- Use `IMAGE_PROCESSING_MODE = "gemini"` for technical diagrams

**3. Local LLM Performance:**
- **GPU**: Ollama auto-detects NVIDIA GPUs (check with `nvidia-smi`)
- **CPU**: Use `phi3:mini` or `llama3.2:1b` for decent speed
- **Timeout**: 180 seconds default (configurable in `qa_engine.py`)

### Resource Usage

| Task | CPU | RAM | Time (Approx) |
|------|-----|-----|---------------|
| First PDF processing (50 pages) | Medium | 2-4GB | 30-60s |
| Cached PDF loading | Low | 1-2GB | 3-5s |
| Question answering (Gemini) | Low | 1-2GB | 1-3s |
| Question answering (Local, phi3:mini) | High | 3-5GB | 20-120s |
| Embedding generation | Medium | 2-3GB | Per page: 0.5-1s |

---

## 🔧 Troubleshooting

### Common Issues

**1. "Timeout exceeded" with Local LLM**

**Problem:** CPU model is too slow (>180 seconds)

**Solutions:**
- Switch to faster model:
  ```bash
  ollama pull phi3:mini
  # Edit config/config.py: LOCAL_LLM_MODEL = "phi3:mini"
  ```
- Use GPU if available (check `nvidia-smi`)
- Reduce context: `TOP_K_RESULTS = 2` in `config/config.py`
- Switch to cloud LLM: `LLM_TYPE = "gemini"`

**2. "Cannot connect to Ollama"**

**Problem:** Ollama server not running

**Solution:**
```bash
ollama serve
```

Verify with:
```bash
ollama list
```

**3. "Turkish characters broken" in OCR**

**Problem:** Tesseract missing Turkish language pack

**Solution:**
- Windows: Reinstall Tesseract and select "Turkish" language pack
- Linux: `sudo apt install tesseract-ocr-tur`
- macOS: `brew install tesseract-lang`

**4. "Poor OCR quality"**

**Solutions:**
- Ensure high-quality PDF (not scanned at low DPI)
- Switch to Gemini Vision: `IMAGE_PROCESSING_MODE = "gemini"`
- Adjust preprocessing in `src/image_processor.py` (increase upscale target)

**5. "Out of memory"**

**Solutions:**
- Reduce `CHUNK_SIZE` and `TOP_K_RESULTS`
- Use smaller local model (`llama3.2:1b`)
- Process PDFs in smaller batches
- Close other applications

---

## 📁 Data Management

### Vector Database Files

**Location:** `data/vectordb/`

**Files:**
- `index.faiss` - FAISS vector index (embeddings)
- `documents.pkl` - Document chunks and metadata

**When to Clear:**
- PDF content changed
- Configuration changes (chunk size, overlap)
- Switching embedding models
- Corruption or errors

**How to Clear:**
- CLI: Run `python main.py` → Option 4
- Manual: Delete `data/vectordb/` folder
- Web UI: Settings → Clear Database

### Uploaded PDFs

**Location:** `data/uploaded/`

Manually manage this folder - files are kept for reference but not automatically cleaned.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Open an issue with detailed reproduction steps
2. **Suggest Features**: Describe your use case and proposed solution
3. **Improve Documentation**: Fix typos, add examples, translate
4. **Submit Code**: Fork, create feature branch, submit PR

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/smart-manual.git
cd smart-manual
pip install -r requirements.txt

# Run tests (if available)
pytest tests/

# Code style
black src/ app.py main.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**Technologies & Libraries:**
- [Google Gemini](https://ai.google.dev/) - Powerful multimodal AI
- [Ollama](https://ollama.ai/) - Easy local LLM deployment
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook's vector similarity search
- [Sentence Transformers](https://www.sbert.net/) - State-of-the-art embeddings
- [Streamlit](https://streamlit.io/) - Beautiful Python web apps
- [pdfplumber](https://github.com/jsvine/pdfplumber) - Robust PDF parsing
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Open-source OCR engine

**Inspiration:**
- Built to solve real-world documentation challenges in industrial environments
- Designed with privacy and cost-effectiveness as core principles

---

## 📞 Support

**Issues & Questions:**
- GitHub Issues: [Report a problem](https://github.com/yourusername/smart-manual/issues)
- Discussions: [Ask questions](https://github.com/yourusername/smart-manual/discussions)

**Detailed Setup:**
- See [SETUP_GUIDE.md](SETUP_GUIDE.md) for step-by-step installation instructions

---

## 🗺️ Roadmap

**Planned Features:**
- [ ] Multi-PDF support (query across multiple manuals)
- [ ] Conversation history and follow-up questions
- [ ] PDF annotation export (save Q&A to PDF)
- [ ] REST API for integration
- [ ] Docker containerization
- [ ] Streaming responses for local LLM
- [ ] Additional embedding model options
- [ ] Automatic language detection

---

## 📊 Project Stats

- **Language:** Python 3.8+
- **Lines of Code:** ~2,000
- **Architecture:** Modular, production-ready
- **License:** MIT
- **Status:** Active development

---

<div align="center">

**Built with ❤️ for better documentation experiences**

[⭐ Star on GitHub](https://github.com/yourusername/smart-manual) · [🐛 Report Bug](https://github.com/yourusername/smart-manual/issues) · [💡 Request Feature](https://github.com/yourusername/smart-manual/issues)

</div>
