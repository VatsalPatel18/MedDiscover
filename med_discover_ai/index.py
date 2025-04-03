# med_discover_ai/index.py
import faiss
import numpy as np
import os
from med_discover_ai.config import INDEX_SAVE_PATH, EMBEDDING_DIMENSION, USE_GPU

def build_faiss_index(embeddings):
    """
    Build a FAISS index from the given embeddings.

    Parameters:
        embeddings (np.array): Array of embeddings (shape [N, D]).

    Returns:
        faiss.Index or None: FAISS index if successful, None otherwise.
    """
    if embeddings is None or embeddings.shape[0] == 0:
        print("Error: Cannot build index from empty or invalid embeddings.")
        return None

    dimension = embeddings.shape[1]

    # Verify dimension consistency
    expected_dim = EMBEDDING_DIMENSION
    if dimension != expected_dim:
        print(f"Warning: Embedding dimension mismatch. Expected {expected_dim}, got {dimension}.")
        # Decide how to handle: error out, or proceed with caution?
        # For now, proceed but log warning. If critical, raise an error.

    print(f"Building FAISS index with dimension {dimension}...")

    try:
        # Using Inner Product (IP) index for MedCPT as recommended by original paper/examples.
        # For OpenAI embeddings (like ada-002), L2 (Euclidean distance) is more common.
        # We choose based on the mode.
        if USE_GPU:
            # MedCPT typically uses Inner Product (cosine similarity after normalization)
            print("Using IndexFlatIP (Inner Product) for MedCPT embeddings.")
            index = faiss.IndexFlatIP(dimension)
        else:
            # OpenAI embeddings often use L2 (Euclidean distance)
            print("Using IndexFlatL2 (Euclidean Distance) for OpenAI embeddings.")
            index = faiss.IndexFlatL2(dimension)

        # FAISS requires float32 type
        if embeddings.dtype != np.float32:
            print("Converting embeddings to float32 for FAISS.")
            embeddings = embeddings.astype(np.float32)

        # Add embeddings to the index
        index.add(embeddings)
        print(f"FAISS index built successfully with {index.ntotal} vectors.")
        return index
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        return None

def save_index(index, path=INDEX_SAVE_PATH):
    """
    Save the FAISS index to disk.

    Parameters:
        index (faiss.Index): The FAISS index to save.
        path (str): The file path to save the index to.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if index is None:
        print("Error: Cannot save a null index.")
        return False
    try:
        print(f"Saving FAISS index to {path}...")
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(index, path)
        print("Index saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving FAISS index to {path}: {e}")
        return False

def load_index(path=INDEX_SAVE_PATH):
    """
    Load the FAISS index from disk.

    Parameters:
        path (str): The file path to load the index from.

    Returns:
        faiss.Index or None: The loaded FAISS index, or None if loading fails.
    """
    if not os.path.exists(path):
        print(f"Error: Index file not found at {path}.")
        return None
    try:
        print(f"Loading FAISS index from {path}...")
        index = faiss.read_index(path)
        print(f"Index loaded successfully with {index.ntotal} vectors and dimension {index.d}.")
        # Optional: Verify dimension against config
        if index.d != EMBEDDING_DIMENSION:
             print(f"Warning: Loaded index dimension ({index.d}) differs from config ({EMBEDDING_DIMENSION}).")
        return index
    except Exception as e:
        print(f"Error loading FAISS index from {path}: {e}")
        return None

