# RAG Banking Chatbot

A production-ready AI-powered banking chatbot using Retrieval Augmented Generation (RAG) with local LLM and vector database.

## 🎯 Features

- **Local AI Processing**: Uses Ollama with Llama 3 model - no external API calls
- **RAG Architecture**: Retrieval Augmented Generation for accurate banking information
- **PDF Document Ingestion**: Upload and process banking policy documents
- **Vector Database**: ChromaDB for efficient semantic search
- **Banking-Specific Prompts**: Specialized responses for different query types
- **Privacy-Focused**: All processing happens locally
- **Compliance Ready**: Includes security disclaimers and fraud alerts
- **Modern UI**: Two-panel layout with real-time chat interface

## 🏗️ Project Structure

```
rag-banking-chatbot/
├── app.py                 # FastAPI application
├── pdf_processor.py       # PDF extraction and chunking
├── rag_engine.py          # RAG implementation with Ollama
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Web interface
├── ingested_documents/    # Document metadata storage
└── chroma_db/            # Vector database storage
```

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- 8GB+ RAM (recommended)
- Ollama installed and running

### Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai)

## 🚀 Quick Start

### 1. Start Ollama Service

```bash
# Start Ollama (in a separate terminal)
ollama serve

# The service will run on http://localhost:11434
# First time will automatically pull llama3:latest
```

### 2. Setup Python Environment

```bash
# Clone repository
git clone https://github.com/yourusername/rag-banking-chatbot.git
cd rag-banking-chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start the Application

```bash
python app.py
```

The application will start on `http://localhost:8000`

### 4. Use the Chat Interface

1. Open your browser to `http://localhost:8000`
2. Upload a PDF document using the sidebar
3. Ask questions about the document
4. Select query type (General, Account, Loan, Security) for specialized responses

## 🔌 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Upload PDF
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload-pdf
```

### Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the account opening requirements?",
    "query_type": "general",
    "temperature": 0.3
  }'
```

### List Documents
```bash
curl http://localhost:8000/documents
```

### Delete Document
```bash
curl -X DELETE http://localhost:8000/documents/{document_id}
```

### Get Statistics
```bash
curl http://localhost:8000/stats
```

## 🔧 Configuration

### Customize RAG Engine (in app.py)

```python
rag_engine = RAGEngine(
    embedding_model="all-MiniLM-L6-v2",  # Sentence transformer model
    ollama_model="llama3:latest",         # Ollama model
    db_path="./chroma_db",                # Vector DB location
    top_k=4                                # Top-K retrieval
)
```

### Customize PDF Processing (in pdf_processor.py)

```python
pdf_processor = PDFProcessor(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap between chunks
)
```

## 📝 Query Types

### 1. General Banking
- General questions about banking services
- Policy inquiries
- General guidance

### 2. Account Queries
- Account balance information
- Account services
- Account statements

### 3. Loan Inquiries
- Loan terms and rates
- Eligibility requirements
- Application process
- Includes rate and terms disclaimer

### 4. Security/Fraud
- Security alerts and warnings
- Fraud prevention
- Suspicious activity reporting
- Emphasizes never asking for passwords

## 🛡️ Security Features

- **Local Processing**: No data sent to external APIs
- **Compliance Disclaimers**: Automatic disclaimers in responses
- **Security Prompts**: Special handling for security queries
- **Authentication Reminders**: Never asks for sensitive credentials
- **Fraud Detection Alerts**: Security warnings in responses

## 📚 Document Processing Pipeline

1. **Upload**: PDF file uploaded via web interface
2. **Extract**: Text extracted from all pages
3. **Chunk**: Text split into 1000-char chunks with 200-char overlap
4. **Embed**: Chunks converted to embeddings using all-MiniLM-L6-v2
5. **Store**: Embeddings and metadata stored in ChromaDB
6. **Retrieve**: Relevant chunks retrieved for user queries
7. **Generate**: LLM generates response using retrieved context

## 🎓 Example Banking Documents

The system works best with documents like:
- Account opening policies
- Loan application guidelines
- Fraud prevention policies
- Security procedures
- Fee schedules
- Terms and conditions
- Customer service guidelines

## 🔍 Troubleshooting

### Ollama Connection Error
```
Error: Cannot connect to Ollama at http://localhost:11434
```
**Solution**: Start Ollama service
```bash
ollama serve
```

### Model Download Timeout
```
Error: Failed to pull model
```
**Solution**: Manually pull the model
```bash
ollama pull llama3:latest
```

### Out of Memory
**Solution**: Reduce model size
```python
rag_engine = RAGEngine(ollama_model="llama2:latest")  # Smaller model
```

### Slow Response
**Solution**: 
- Reduce chunk size in pdf_processor.py
- Reduce top_k retrieval count
- Use a smaller embedding model

## 📦 Dependencies

- **FastAPI**: Web framework
- **ChromaDB**: Vector database
- **sentence-transformers**: Embedding model
- **pdfplumber**: PDF processing
- **ollama**: LLM integration
- **torch**: Deep learning framework

## 🚀 Advanced Usage

### Batch PDF Processing

```bash
python pdf_processor.py banking_policy.pdf "Banking Policy"
```

### Custom Prompt Templates

Edit the `_init_prompts()` method in `rag_engine.py`:

```python
def _init_prompts(self) -> dict:
    return {
        "custom_type": "Your custom prompt template here"
    }
```

### Integration with Existing Systems

The API can be integrated with:
- Customer service platforms
- Mobile banking apps
- Internal knowledge bases
- Chatbot frameworks

## 📈 Performance Optimization

1. **Embeddings Caching**: Responses are cached after first query
2. **Batch Processing**: Multiple PDFs processed efficiently
3. **Vector DB Optimization**: ChromaDB uses HNSW for fast search
4. **Temperature Control**: Adjustable for accuracy vs diversity

## 🔐 Privacy & Compliance

- ✅ GDPR compliant (local processing)
- ✅ No data transmitted externally
- ✅ Automatic compliance disclaimers
- ✅ Security-focused prompts
- ✅ Fraud prevention reminders

## 📋 Logging

All operations are logged to console:
- Document uploads
- Vector DB operations
- LLM queries
- Error tracking

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional query types
- Language support
- Database optimization
- UI enhancements

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues:
1. Check troubleshooting section above
2. Review API logs in console
3. Verify Ollama is running
4. Ensure documents are uploaded

## 🎓 Learning Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/)
- [RAG Concepts](https://arxiv.org/abs/2005.11401)

## ⚠️ Limitations

- Requires 8GB+ RAM for Llama 3
- Response time depends on chunk size and model
- Accuracy limited by document quality
- Single-turn conversations (no memory)

## 🚀 Future Enhancements

- [ ] Multi-turn conversation memory
- [ ] Streaming responses
- [ ] Document OCR support
- [ ] Multi-language support
- [ ] Advanced analytics
- [ ] User authentication
- [ ] Document versioning

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready ✅