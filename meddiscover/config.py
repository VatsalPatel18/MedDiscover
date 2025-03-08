import os
import torch

# Determine if GPU is available.
USE_GPU = torch.cuda.is_available()

if USE_GPU:
    # GPU mode: use MedCPT models.
    ARTICLE_ENCODER_MODEL = "ncbi/MedCPT-Article-Encoder"
    QUERY_ENCODER_MODEL = "ncbi/MedCPT-Query-Encoder"
    CROSS_ENCODER_MODEL = "ncbi/MedCPT-Cross-Encoder"
    # Define EMBEDDING_MODEL even if it won’t be used in GPU branch.
    EMBEDDING_MODEL = "text-embedding-ada-002"
else:
    # CPU mode: use OpenAI's embedding model.
    EMBEDDING_MODEL = "text-embedding-ada-002"

# Common configuration parameters
CHUNK_SIZE = 500
OVERLAP = 50
MAX_ARTICLE_LENGTH = 512
MAX_QUERY_LENGTH = 64

DEFAULT_PDF_FOLDER = "./sample_pdf_rag"
INDEX_SAVE_PATH = "./faiss_index.bin"
DOC_META_PATH = "./doc_metadata.json"

# OpenAI API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# LLM model for answer generation
LLM_MODEL = "gpt-4"
