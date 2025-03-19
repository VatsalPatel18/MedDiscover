from med_discover_ai.config import USE_GPU, CROSS_ENCODER_MODEL
if USE_GPU:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    cross_tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
    cross_model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL).to("cuda")
    cross_model.eval()
else:
    cross_tokenizer = None
    cross_model = None

import torch
import numpy as np
from med_discover_ai.embeddings import embed_query
from med_discover_ai.index import load_index
import json
from med_discover_ai.config import DOC_META_PATH

def load_metadata(meta_path=DOC_META_PATH):
    """
    Load document metadata from a JSON file.
    
    Returns:
        list: List of document metadata dictionaries.
    """
    with open(meta_path, "r") as f:
        return json.load(f)

def rerank(query, candidates):
    """
    Re-rank candidates using the MedCPT Cross Encoder if GPU is available,
    otherwise return retrieval scores.
    
    Parameters:
        query (str): The user query.
        candidates (list): List of candidate dictionaries with 'text' and 'retrieval_score'.
    
    Returns:
        list or np.array: Relevance scores for each candidate.
    """
    if USE_GPU:
        pairs = [[query, candidate["text"]] for candidate in candidates]
        with torch.no_grad():
            encoded = cross_tokenizer(pairs, truncation=True, padding=True, return_tensors="pt", max_length=512)
            for k in encoded:
                encoded[k] = encoded[k].to("cuda")
            logits = cross_model(**encoded).logits.squeeze(dim=1)
        print('re-ranked with MedCPT')
        return logits.cpu().numpy()
    else:
        print('no re-rank performed')
        return [candidate["retrieval_score"] for candidate in candidates]

def search_with_rerank(query, index, doc_metadata, k=5):
    """
    Retrieve and re-rank the top-k candidates for a given query using MedCPT encoders when GPU is available.
    
    Parameters:
        query (str): The user query.
        index (faiss.Index): The FAISS index.
        doc_metadata (list): List of document metadata.
        k (int): Number of top results to return.
    
    Returns:
        list: Candidate documents sorted by re-rank scores (Cross Encoder if GPU, retrieval scores if CPU).
    """
    # Step 1: Dense retrieval using FAISS
    query_embedding = embed_query(query)  # shape [1, embedding_dim]
    scores, inds = index.search(query_embedding, k)
    
    candidates = []
    for score, ind in zip(scores[0], inds[0]):
        entry = doc_metadata[ind].copy()  # Avoid modifying original metadata
        entry["retrieval_score"] = float(score)
        candidates.append(entry)
    
    # Step 2: Re-ranking
    rerank_scores = rerank(query, candidates)
    for i, score in enumerate(rerank_scores):
        candidates[i]["rerank_score"] = float(score)
    
    # Sort by re-rank scores
    candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    print('searched with re-rank')
    return candidates