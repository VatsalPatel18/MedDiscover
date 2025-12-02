# med_discover_ai/config.py
import os
import torch

# --- Core Settings ---
# Determine if GPU is available. This affects default model choices and execution paths.
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
ALLOW_MEDCPT_CPU = os.environ.get("ALLOW_MEDCPT_CPU", "0") == "1"
print(f"GPU Available: {USE_GPU}, Using device: {DEVICE}")

# --- Available Models ---
# Define available embedding models for the UI dropdown
AVAILABLE_EMBEDDING_MODELS = {
    "MedCPT (GPU Recommended)": "ncbi/MedCPT-Article-Encoder", # Use this identifier internally
    "OpenAI Ada-002 (CPU/Cloud)": "text-embedding-ada-002"     # Use this identifier internally
}

# Define available LLM models for the UI dropdown
# Combine OpenAI and Ollama models
AVAILABLE_LLM_MODELS = [
    # OpenAI Models (only 4.1 variants retained)
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    # Ollama Models (Prefix with 'ollama:' for easy identification)
    "ollama:gemma3:1b", # Added Gemma3 1B
    "ollama:gemma3:4b", # Added Gemma3 4B
    "ollama:llama3:8b", # Kept standard Llama3 8B
    "ollama:phi3:mini",
]

# --- Default Model Configuration (Based on GPU availability) ---
if USE_GPU or ALLOW_MEDCPT_CPU:
    print('GPU detected. Setting defaults for GPU usage.' if USE_GPU else 'GPU not available, CPU override enabled for MedCPT.')
    DEFAULT_EMBEDDING_MODEL_NAME = "MedCPT (GPU Recommended)"
    DEFAULT_LLM_MODEL = "gpt-4.1-mini"

    ARTICLE_ENCODER_MODEL = AVAILABLE_EMBEDDING_MODELS["MedCPT (GPU Recommended)"]
    QUERY_ENCODER_MODEL = "ncbi/MedCPT-Query-Encoder"
    CROSS_ENCODER_MODEL = "ncbi/MedCPT-Cross-Encoder"
    EMBEDDING_DIMENSION = 768 # MedCPT embedding dimension
    OPENAI_EMBEDDING_MODEL_ID = AVAILABLE_EMBEDDING_MODELS["OpenAI Ada-002 (CPU/Cloud)"]

else:
    print('GPU not available. Setting defaults for CPU usage.')
    DEFAULT_EMBEDDING_MODEL_NAME = "OpenAI Ada-002 (CPU/Cloud)"
    DEFAULT_LLM_MODEL = "ollama:gemma3:4b"

    ARTICLE_ENCODER_MODEL = None
    QUERY_ENCODER_MODEL = None
    CROSS_ENCODER_MODEL = None # Re-ranking is disabled on CPU anyway
    OPENAI_EMBEDDING_MODEL_ID = AVAILABLE_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL_NAME]
    EMBEDDING_DIMENSION = 1536 # Dimension for text-embedding-ada-002

# --- Text Processing Parameters ---
CHUNK_SIZE = 500 # Number of words per chunk
OVERLAP = 50     # Number of overlapping words between chunks
MAX_ARTICLE_LENGTH = 512 # Max tokens for MedCPT article/cross encoder input
MAX_QUERY_LENGTH = 64    # Max tokens for MedCPT query encoder input

# --- File Paths ---
DEFAULT_PDF_FOLDER = "./sample_pdf_rag"
INDEX_SAVE_PATH = "./faiss_index.bin"
DOC_META_PATH = "./doc_metadata.json"

# --- API Configuration ---
# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE":
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print("OpenAI API Key found in environment.")
else:
    print("Warning: OpenAI API Key not found in environment variables. Please set it in the UI if using OpenAI models.")

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
print(f"Using Ollama base URL: {OLLAMA_BASE_URL}")

# --- LLM Inference Configuration ---
DEFAULT_MAX_TOKENS = 150 # Default max tokens for LLM response generation

# --- Retrieval Configuration ---
DEFAULT_K = 5 # Default number of chunks to retrieve
# Re-ranking is only attempted if USE_GPU is True AND the checkbox is enabled in UI
DEFAULT_RERANK_ENABLED = True if USE_GPU else False # Default based on GPU, but UI controls final decision

# --- Helper function to get internal model ID from display name ---
def get_embedding_model_id(display_name):
    """Gets the internal model identifier from the display name."""
    return AVAILABLE_EMBEDDING_MODELS.get(display_name)

def get_embedding_dimension(display_name):
    """Gets the embedding dimension based on the display name."""
    model_id = get_embedding_model_id(display_name)
    if model_id == "ncbi/MedCPT-Article-Encoder":
        return 768
    elif model_id == "text-embedding-ada-002":
        return 1536
    else:
        # Fallback or default dimension if unknown model selected
        print(f"Warning: Unknown embedding model '{display_name}', returning default dimension 768.")
        return 768 # Or choose a more appropriate default/error handling
