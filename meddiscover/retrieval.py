from embeddings import embed_query
from index import load_index
import json
from config import DOC_META_PATH

def load_metadata(meta_path=DOC_META_PATH):
    """
    Load document metadata from a JSON file.
    """
    with open(meta_path, "r") as f:
        return json.load(f)

def search_with_rerank(query, index, doc_metadata, k=5):
    """
    Retrieve and re-rank the top-k candidates for a given query.
    
    Parameters:
        query (str): The user query.
        index (faiss.Index): The FAISS index.
        doc_metadata (list): List of document metadata.
        k (int): Number of top results to return.
    
    Returns:
        list: Candidate documents with scores.
    """
    # Step 1: Dense retrieval using FAISS.
    query_embedding = embed_query(query)  # shape [1, embedding_dim]
    scores, inds = index.search(query_embedding, k)
    
    candidates = []
    for score, ind in zip(scores[0], inds[0]):
        entry = doc_metadata[ind]
        entry["retrieval_score"] = float(score)
        candidates.append(entry)
    
    # Step 2: (Optional) Re-ranking with a cross encoder can be added here.
    # For now, we simply use the retrieval score.
    for candidate in candidates:
        candidate["rerank_score"] = candidate["retrieval_score"]
    
    candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return candidates
