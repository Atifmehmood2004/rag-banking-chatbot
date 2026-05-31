"""
PDF Processor Module
Handles PDF text extraction, chunking, and preprocessing for RAG.
"""

import os
import logging
from typing import List, Tuple
from pathlib import Path
import pdfplumber
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Process PDF documents for RAG pipeline.
    Extracts text, chunks content, and manages document metadata.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents_dir = Path("ingested_documents")
        self.documents_dir.mkdir(exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text from PDF
            
        Raises:
            FileNotFoundError: If PDF file not found
            Exception: If PDF extraction fails
        """
        try:
            logger.info(f"Extracting text from: {pdf_path}")
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n{page_text}"
            
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise

    def chunk_text(self, text: str, document_id: str) -> List[Tuple[str, dict]]:
        """
        Split text into overlapping chunks with metadata.
        
        Args:
            text: Text to chunk
            document_id: ID of the source document
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        chunks = []
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(text):
            end_idx = start_idx + self.chunk_size
            chunk = text[start_idx:end_idx]
            
            metadata = {
                "document_id": document_id,
                "chunk_index": chunk_num,
                "start_char": start_idx,
                "end_char": min(end_idx, len(text))
            }
            
            chunks.append((chunk, metadata))
            chunk_num += 1
            start_idx = end_idx - self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

    def process_pdf(self, pdf_path: str, document_name: str) -> Tuple[List[Tuple[str, dict]], dict]:
        """
        Complete pipeline: extract and chunk PDF.
        
        Args:
            pdf_path: Path to PDF file
            document_name: Name for the document
            
        Returns:
            Tuple of (chunks, document_metadata)
        """
        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            # Generate document ID
            document_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(document_name).stem}"
            
            # Create chunks
            chunks = self.chunk_text(text, document_id)
            
            # Create metadata
            metadata = {
                "document_id": document_id,
                "original_name": document_name,
                "upload_date": datetime.now().isoformat(),
                "total_chunks": len(chunks),
                "total_characters": len(text),
                "file_path": pdf_path
            }
            
            # Save document info
            self._save_document_info(metadata)
            
            return chunks, metadata
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def _save_document_info(self, metadata: dict) -> None:
        """
        Save document metadata to file.
        
        Args:
            metadata: Document metadata
        """
        import json
        doc_info_path = self.documents_dir / f"{metadata['document_id']}.json"
        with open(doc_info_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved document info: {doc_info_path}")

    def get_document_list(self) -> List[dict]:
        """
        Get list of all ingested documents.
        
        Returns:
            List of document metadata dictionaries
        """
        import json
        documents = []
        for json_file in self.documents_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    doc_meta = json.load(f)
                    documents.append(doc_meta)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse document info: {json_file}")
        return documents

    def delete_document_info(self, document_id: str) -> bool:
        """
        Delete document metadata.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if deleted successfully
        """
        doc_info_path = self.documents_dir / f"{document_id}.json"
        if doc_info_path.exists():
            doc_info_path.unlink()
            logger.info(f"Deleted document info: {document_id}")
            return True
        return False

    def clean_temp_file(self, file_path: str) -> None:
        """
        Clean up temporary PDF file.
        
        Args:
            file_path: Path to file to delete
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up temp file: {str(e)}")


# CLI Interface for batch processing
if __name__ == "__main__":
    import sys
    
    processor = PDFProcessor()
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        doc_name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(pdf_file)
        
        try:
            chunks, metadata = processor.process_pdf(pdf_file, doc_name)
            print(f"✓ Successfully processed: {doc_name}")
            print(f"  Document ID: {metadata['document_id']}")
            print(f"  Total chunks: {metadata['total_chunks']}")
            print(f"  Total characters: {metadata['total_characters']}")
        except Exception as e:
            print(f"✗ Error processing PDF: {str(e)}")
            sys.exit(1)
    else:
        print("Usage: python pdf_processor.py <pdf_path> [document_name]")
        sys.exit(1)