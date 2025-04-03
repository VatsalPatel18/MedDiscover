# med_discover_ai/config.py
import os
import torch

# --- Core Settings ---
# Determine if GPU is available for MedCPT models.
USE_GPU = torch.cuda.is_available()

# --- Model Configuration ---
if USE_GPU:
    print('GPU is available. Using MedCPT models.')
    # GPU mode: use MedCPT models.
    ARTICLE_ENCODER_MODEL = "ncbi/MedCPT-Article-Encoder"
    QUERY_ENCODER_MODEL = "ncbi/MedCPT-Query-Encoder"
    CROSS_ENCODER_MODEL = "ncbi/MedCPT-Cross-Encoder"
    EMBEDDING_MODEL = None # Not used in GPU mode for primary embedding
    EMBEDDING_DIMENSION = 768 # MedCPT embedding dimension
else:
    print('GPU not available. Using OpenAI embeddings (CPU mode).')
    # CPU mode: use OpenAI's embedding model.
    ARTICLE_ENCODER_MODEL = None
    QUERY_ENCODER_MODEL = None
    CROSS_ENCODER_MODEL = None
    EMBEDDING_MODEL = "text-embedding-ada-002" # Default OpenAI model
    EMBEDDING_DIMENSION = 1536 # Dimension for text-embedding-ada-002

# --- Text Processing Parameters ---
CHUNK_SIZE = 500 # Number of words per chunk
OVERLAP = 50     # Number of overlapping words between chunks
MAX_ARTICLE_LENGTH = 512 # Max tokens for article/cross encoder input
MAX_QUERY_LENGTH = 64     # Max tokens for query encoder input

# --- File Paths ---
DEFAULT_PDF_FOLDER = "./sample_pdf_rag" # Default location for input PDFs if needed
INDEX_SAVE_PATH = "./faiss_index.bin" # Path to save/load the FAISS index
DOC_META_PATH = "./doc_metadata.json" # Path to save/load document metadata

# --- OpenAI API Configuration ---
# Load API key from environment variable first, then fallback to a placeholder.
# IMPORTANT: It's best practice to set the API key via environment variables.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE":
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print("OpenAI API Key found.")
else:
    print("Warning: OpenAI API Key not found in environment variables. Please set it in the UI.")

# --- LLM Configuration ---
# List of available LLM models for the UI dropdown
AVAILABLE_LLM_MODELS = ["gpt-4o", "gpt-4o-mini-2024-07-18"]
# Default LLM model to use for answer generation
DEFAULT_LLM_MODEL = "gpt-4o"
# Default max tokens for LLM response generation
DEFAULT_MAX_TOKENS = 75 # Increased default slightly

# --- Retrieval Configuration ---
DEFAULT_K = 3 # Default number of chunks to retrieve (Changed from 5)
DEFAULT_RERANK_ENABLED = True # Default state for enabling/disabling re-ranking (primarily affects GPU mode)

# --- Device Configuration ---
DEVICE = "cuda" if USE_GPU else "cpu"
print(f"Using device: {DEVICE}")

