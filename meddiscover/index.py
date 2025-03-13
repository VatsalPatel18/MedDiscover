import faiss
import numpy as np
from meddiscover.config import INDEX_SAVE_PATH

def build_faiss_index(embeddings):
    """
    Build a FAISS index from the given embeddings.
    
    Parameters:
        embeddings (np.array): Array of embeddings.
        
    Returns:
        faiss.Index: FAISS index.
    """
    dimension = embeddings.shape[1]
    # Using inner product (IP) index; adjust to L2 if needed.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def save_index(index, path=INDEX_SAVE_PATH):
    """Save the FAISS index to disk."""
    faiss.write_index(index, path)

def load_index(path=INDEX_SAVE_PATH):
    """Load the FAISS index from disk."""
    return faiss.read_index(path)
