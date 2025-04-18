# med_discover_ai/index.py
import faiss
import numpy as np
import os
import torch # Keep torch import for USE_GPU check if needed, though logic changes
from med_discover_ai.config import INDEX_SAVE_PATH
# Removed EMBEDDING_DIMENSION and USE_GPU import as direct dependencies for logic here
# Dimension and index type will be determined dynamically or passed in.

def build_faiss_index(embeddings, use_ip_metric=None):
    """
    Build a FAISS index from the given embeddings, choosing metric dynamically.

    Parameters:
        embeddings (np.array): Array of embeddings (shape [N, D]).
        use_ip_metric (bool, optional): Explicitly set to True for Inner Product (MedCPT),
                                        False for L2 (OpenAI). If None, uses dimension
                                        heuristic (768=IP, others=L2).

    Returns:
        faiss.Index or None: FAISS index if successful, None otherwise.
    """
    if embeddings is None or embeddings.shape[0] == 0:
        print("Error: Cannot build index from empty or invalid embeddings.")
        return None

    dimension = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dimension}...")

    try:
        # Determine Index Type (Metric)
        if use_ip_metric is True:
            print("Using IndexFlatIP (Inner Product) as explicitly requested.")
            index = faiss.IndexFlatIP(dimension)
        elif use_ip_metric is False:
            print("Using IndexFlatL2 (Euclidean Distance) as explicitly requested.")
            index = faiss.IndexFlatL2(dimension)
        else:
            # Heuristic based on dimension if not specified
            if dimension == 768: # Likely MedCPT
                print("Using IndexFlatIP (Inner Product) based on dimension 768 (heuristic for MedCPT).")
                index = faiss.IndexFlatIP(dimension)
            else: # Assume L2 for others (like OpenAI 1536)
                print(f"Using IndexFlatL2 (Euclidean Distance) based on dimension {dimension} (heuristic for non-MedCPT).")
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
    Save the FAISS index to disk. (No changes needed here)

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

def load_index(path=INDEX_SAVE_PATH, expected_dim=None):
    """
    Load the FAISS index from disk, optionally checking its dimension.

    Parameters:
        path (str): The file path to load the index from.
        expected_dim (int, optional): If provided, checks if the loaded index's
                                      dimension matches this value.

    Returns:
        faiss.Index or None: The loaded FAISS index, or None if loading fails or
                             dimension mismatch occurs (if expected_dim is checked).
    """
    if not os.path.exists(path):
        print(f"Error: Index file not found at {path}.")
        return None
    try:
        print(f"Loading FAISS index from {path}...")
        index = faiss.read_index(path)
        print(f"Index loaded successfully with {index.ntotal} vectors and dimension {index.d}.")

        # Verify dimension if expected_dim is provided
        if expected_dim is not None and index.d != expected_dim:
             # This is a critical error, as using the wrong model with the index will fail.
             print(f"FATAL ERROR: Loaded index dimension ({index.d}) differs from expected dimension ({expected_dim}).")
             print("This indicates a mismatch between the loaded index and the currently selected embedding model.")
             print("Please ensure the correct embedding model is selected or re-process PDFs with that model.")
             return None # Return None on dimension mismatch

        return index
    except Exception as e:
        print(f"Error loading FAISS index from {path}: {e}")
        return None

# Optional: Function to load metadata separately if needed elsewhere
# (Already present in retrieval.py, maybe keep it consolidated there)
# def load_index_metadata(meta_path=DOC_META_PATH):
#     # ... implementation ...
