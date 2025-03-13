import numpy as np
import torch
from med_discover_ai.config import USE_GPU, ARTICLE_ENCODER_MODEL, EMBEDDING_MODEL, MAX_ARTICLE_LENGTH
from openai import OpenAI

if USE_GPU:
    from transformers import AutoTokenizer, AutoModel
    # Load MedCPT Article Encoder for GPU users.
    article_tokenizer = AutoTokenizer.from_pretrained(ARTICLE_ENCODER_MODEL)
    article_model = AutoModel.from_pretrained(ARTICLE_ENCODER_MODEL).to("cuda")
    article_model.eval()
else:
    import openai

def embed_documents(doc_chunks, batch_size=8):
    """
    Generate embeddings for document chunks.
    
    Parameters:
        doc_chunks (list): List of text chunks.
        batch_size (int): Batch size for processing.
        
    Returns:
        np.array: Array of embeddings.
    """
    if USE_GPU:
        all_embeds = []
        for i in range(0, len(doc_chunks), batch_size):
            batch = doc_chunks[i:i+batch_size]
            with torch.no_grad():
                encoded = article_tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=MAX_ARTICLE_LENGTH)
                for key in encoded:
                    encoded[key] = encoded[key].to("cuda")
                outputs = article_model(**encoded).last_hidden_state[:, 0, :]
                all_embeds.append(outputs.cpu().numpy())
        if all_embeds:
            return np.vstack(all_embeds)
        else:
            return np.array([])
    else:
        # For CPU users, call the OpenAI embedding API.
        embeddings = []
        client = OpenAI()

        for text in doc_chunks:
            response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            embed = response.data[0].embedding
            # embed = response["data"][0].embedding
            embeddings.append(embed)
        return np.array(embeddings)

def embed_query(query, max_query_length=64):
    """
    Generate embedding for a query.
    
    Parameters:
        query (str): Input query.
        max_query_length (int): Maximum token length.
        
    Returns:
        np.array: Query embedding.
    """
    if USE_GPU:
        from transformers import AutoTokenizer, AutoModel
        query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to("cuda")
        query_model.eval()
        with torch.no_grad():
            encoded = query_tokenizer(query, truncation=True, padding=True, return_tensors="pt", max_length=max_query_length)
            for key in encoded:
                encoded[key] = encoded[key].to("cuda")
            outputs = query_model(**encoded).last_hidden_state[:, 0, :]
        return outputs.cpu().numpy()
    else:
        # For CPU, use OpenAI's API for query embeddings.
        client = OpenAI()
        response = client.embeddings.create(input=query, model=EMBEDDING_MODEL)
        embed = response.data[0].embedding
        return np.array([embed])
