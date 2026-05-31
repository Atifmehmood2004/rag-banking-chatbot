"""
FastAPI Application
RESTful API for RAG Banking Chatbot
"""

import logging
import os
import tempfile
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_processor import PDFProcessor
from rag_engine import RAGEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Banking Chatbot",
    description="AI-powered banking assistant with RAG architecture",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
try:
    pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    rag_engine = RAGEngine(
        embedding_model="all-MiniLM-L6-v2",
        ollama_model="llama3:latest",
        db_path="./chroma_db",
        top_k=4
    )
    logger.info("✓ RAG Engine initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize RAG engine: {str(e)}")
    logger.info("Please ensure Ollama is running: ollama serve")
    raise


# ==================== Pydantic Models ====================

class ChatRequest(BaseModel):
    """Chat request model"""
    question: str
    query_type: str = "general"  # general, security, loan, account
    temperature: float = 0.3


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    sources: List[dict]
    query_type: str
    timestamp: str


class DocumentInfo(BaseModel):
    """Document information model"""
    document_id: str
    original_name: str
    upload_date: str
    total_chunks: int
    total_characters: int


class UploadResponse(BaseModel):
    """Upload response model"""
    status: str
    document_id: str
    message: str
    chunks_created: int


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    services: dict
    timestamp: str


# ==================== Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serve the main HTML frontend.
    """
    try:
        with open("templates/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Frontend not found. Please ensure templates/index.html exists.</h1>"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns status of all services.
    """
    from datetime import datetime
    
    try:
        stats = rag_engine.get_collection_stats()
        ollama_status = "healthy"
        pdf_processor_status = "healthy"
    except Exception as e:
        ollama_status = f"error: {str(e)}"
        stats = {}
    
    return HealthResponse(
        status="healthy" if ollama_status == "healthy" else "degraded",
        services={
            "ollama": ollama_status,
            "pdf_processor": pdf_processor_status,
            "vector_db": stats.get("total_documents", 0)
        },
        timestamp=datetime.now().isoformat()
    )


@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload and process PDF document.
    
    Args:
        file: PDF file to upload
        background_tasks: Background tasks executor
        
    Returns:
        Upload response with document ID and chunk count
    """
    try:
        # Validate file type
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Process PDF
            chunks, metadata = pdf_processor.process_pdf(tmp_path, file.filename)
            
            # Add to RAG engine
            rag_engine.add_documents(chunks, metadata["document_id"])
            
            # Clean up temp file
            if background_tasks:
                background_tasks.add_task(pdf_processor.clean_temp_file, tmp_path)
            else:
                pdf_processor.clean_temp_file(tmp_path)
            
            logger.info(f"✓ Successfully processed: {file.filename}")
            
            return UploadResponse(
                status="success",
                document_id=metadata["document_id"],
                message=f"Document '{file.filename}' uploaded and processed successfully",
                chunks_created=metadata["total_chunks"]
            )
        finally:
            # Ensure temp file is cleaned up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat message and generate response using RAG.
    
    Args:
        request: Chat request with question and query type
        
    Returns:
        Chat response with answer and sources
    """
    try:
        from datetime import datetime
        
        # Validate query type
        if request.query_type not in ["general", "security", "loan", "account"]:
            raise HTTPException(status_code=400, detail="Invalid query_type")
        
        # Validate temperature
        if not 0 <= request.temperature <= 1:
            raise HTTPException(status_code=400, detail="Temperature must be between 0 and 1")
        
        # Generate response
        response, sources = rag_engine.generate_response(
            query=request.question,
            query_type=request.query_type,
            temperature=request.temperature
        )
        
        return ChatResponse(
            response=response,
            sources=sources,
            query_type=request.query_type,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """
    List all ingested documents.
    
    Returns:
        List of document information
    """
    try:
        docs = pdf_processor.get_document_list()
        return [
            DocumentInfo(
                document_id=doc["document_id"],
                original_name=doc["original_name"],
                upload_date=doc["upload_date"],
                total_chunks=doc["total_chunks"],
                total_characters=doc["total_characters"]
            )
            for doc in docs
        ]
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Error listing documents")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and its chunks from the vector database.
    
    Args:
        document_id: ID of document to delete
        
    Returns:
        Success message
    """
    try:
        # Delete from vector database
        db_deleted = rag_engine.delete_documents(document_id)
        
        # Delete metadata
        info_deleted = pdf_processor.delete_document_info(document_id)
        
        if db_deleted or info_deleted:
            return JSONResponse(
                status_code=200,
                content={"status": "success", "message": f"Document {document_id} deleted"}
            )
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting document")


@app.get("/stats")
async def get_stats():
    """
    Get statistics about the RAG system.
    
    Returns:
        System statistics
    """
    try:
        collection_stats = rag_engine.get_collection_stats()
        documents = pdf_processor.get_document_list()
        
        return JSONResponse(
            status_code=200,
            content={
                "total_documents": len(documents),
                "total_chunks": collection_stats.get("total_documents", 0),
                "total_characters": sum(doc.get("total_characters", 0) for doc in documents),
                "documents": documents
            }
        )
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting statistics")


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Handle HTTP exceptions.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Handle general exceptions.
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting RAG Banking Chatbot...")
    logger.info("Visit: http://localhost:8000")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )