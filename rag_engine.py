"""
RAG Engine Module
Manages vector database, embeddings, and LLM integration with Ollama.
"""

import logging
from typing import List, Tuple, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of sentence-transformers model
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=False)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.model.encode(texts, convert_to_numpy=False)


class OllamaClient:
    """
    Client for Ollama LLM service.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3:latest"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama service URL
            model: Model to use
        """
        self.base_url = base_url
        self.model = model
        self.api_endpoint = f"{base_url}/api/generate"
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """
        Check if Ollama service is running.
        
        Returns:
            True if service is available
            
        Raises:
            ConnectionError: If Ollama service is not available
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Connected to Ollama service")
                self._ensure_model_available()
                return True
        except requests.exceptions.RequestException:
            logger.error("✗ Cannot connect to Ollama service")
            logger.info("  Please ensure Ollama is running: ollama serve")
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please start Ollama with: ollama serve"
            )
    
    def _ensure_model_available(self) -> None:
        """
        Ensure required model is available, pull if needed.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            if self.model not in model_names:
                logger.info(f"Model {self.model} not found. Attempting to pull...")
                self._pull_model()
            else:
                logger.info(f"✓ Model {self.model} is available")
        except Exception as e:
            logger.warning(f"Could not verify model availability: {str(e)}")
    
    def _pull_model(self) -> None:
        """
        Pull model from Ollama registry.
        
        Raises:
            Exception: If model pull fails
        """
        try:
            logger.info(f"Pulling model: {self.model}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                stream=True,
                timeout=600
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")
                        logger.info(f"  {status}")
                    except json.JSONDecodeError:
                        pass
            
            logger.info(f"✓ Model {self.model} pulled successfully")
        except Exception as e:
            logger.error(f"Failed to pull model: {str(e)}")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            temperature: Temperature for generation (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
            
        Raises:
            Exception: If generation fails
        """
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
        except requests.exceptions.Timeout:
            logger.error("Ollama generation timeout")
            raise
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise


class RAGEngine:
    """
    RAG (Retrieval Augmented Generation) Engine.
    Manages vector database and LLM integration.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        ollama_model: str = "llama3:latest",
        db_path: str = "./chroma_db",
        top_k: int = 4
    ):
        """
        Initialize RAG engine.
        
        Args:
            embedding_model: Sentence-transformer model name
            ollama_model: Ollama model name
            db_path: Path to ChromaDB directory
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k
        
        # Initialize embedding model
        self.embeddings = EmbeddingModel(embedding_model)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None
        self._init_collection()
        
        # Initialize Ollama client
        self.llm = OllamaClient(model=ollama_model)
        
        # Banking prompt templates
        self.prompts = self._init_prompts()
    
    def _init_collection(self) -> None:
        """
        Initialize or get ChromaDB collection.
        """
        try:
            # Delete existing collection to start fresh
            try:
                self.client.delete_collection(name="banking_docs")
                logger.info("Cleared existing collection")
            except:
                pass
            
            # Create new collection with custom embedding function
            self.collection = self.client.get_or_create_collection(
                name="banking_docs",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✓ ChromaDB collection initialized")
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise
    
    def _init_prompts(self) -> dict:
        """
        Initialize banking-specific prompt templates.
        
        Returns:
            Dictionary of prompt templates
        """
        return {
            "general": """You are a professional banking assistant for a financial institution. 
Use ONLY the following retrieved context from our official documents to answer the user's question. 
If the answer is not in the context, say: 'I don't have that information in our documents. Please contact customer service.'
Include security reminders when appropriate.

Context:
{context}

User Question: {question}

Answer:""",
            
            "security": """⚠️ SECURITY ALERT: You are responding to a security-related inquiry.
Emphasis: The bank NEVER asks for passwords, OTPs, or PINs via email, call, or chat.
Provide fraud department contact information and suggest immediate actions if suspicious activity is suspected.

Context:
{context}

User Question: {question}

Answer:""",
            
            "loan": """You are a loan specialist. Focus on: interest rates, terms, eligibility requirements, and application process.
Always add this disclaimer: 'Rates subject to credit approval. Terms and conditions apply.'

Context:
{context}

User Question: {question}

Answer:""",
            
            "account": """You are an account specialist. Help with queries about account balances, statements, and account services.
Remember: We never ask for sensitive credentials via communication channels.

Context:
{context}

User Question: {question}

Answer:"""
        }
    
    def add_documents(self, chunks: List[Tuple[str, dict]], document_id: str) -> None:
        """
        Add document chunks to vector database.
        
        Args:
            chunks: List of (text, metadata) tuples
            document_id: Document ID for grouping
        """
        try:
            logger.info(f"Adding {len(chunks)} chunks to database")
            
            texts = [chunk[0] for chunk in chunks]
            metadatas = [chunk[1] for chunk in chunks]
            ids = [f"{document_id}_{i}" for i in range(len(chunks))]
            
            # Generate embeddings
            embeddings = self.embeddings.embed_batch(texts)
            
            # Add to collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✓ Successfully added {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> Tuple[List[str], List[dict]]:
        """
        Retrieve relevant documents for query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (documents, sources)
        """
        try:
            k = top_k or self.top_k
            
            # Generate query embedding
            query_embedding = self.embeddings.embed(query)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            documents = results.get("documents", [[]])[0] if results.get("documents") else []
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            distances = results.get("distances", [[]])[0] if results.get("distances") else []
            
            # Add similarity scores to metadata
            for metadata, distance in zip(metadatas, distances):
                metadata["similarity"] = 1 - distance  # Convert distance to similarity
            
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents, metadatas
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return [], []
    
    def generate_response(
        self,
        query: str,
        query_type: str = "general",
        temperature: float = 0.3
    ) -> Tuple[str, List[dict]]:
        """
        Generate response using RAG.
        
        Args:
            query: User query
            query_type: Type of query (general, security, loan, account)
            temperature: LLM temperature
            
        Returns:
            Tuple of (response, sources)
        """
        try:
            # Retrieve context
            documents, sources = self.retrieve_context(query)
            
            if not documents:
                return "I don't have any documents in my knowledge base yet. Please upload banking documents first.", []
            
            # Prepare context
            context = "\n\n".join(documents)
            
            # Select prompt template
            prompt_template = self.prompts.get(query_type, self.prompts["general"])
            
            # Format prompt
            prompt = prompt_template.format(
                context=context,
                question=query
            )
            
            # Generate response
            response = self.llm.generate(prompt, temperature=temperature)
            
            # Add compliance disclaimer
            disclaimer = "\n\n📋 **Compliance Note:** This response is based on our banking documents. For official guidance, please contact customer service."
            response += disclaimer
            
            return response, sources
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}", []
    
    def delete_documents(self, document_id: str) -> bool:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Get all IDs for this document
            results = self.collection.get(
                where={"document_id": {"$eq": document_id}}
            )
            ids_to_delete = results.get("ids", [])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for document {document_id}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": "banking_docs",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}