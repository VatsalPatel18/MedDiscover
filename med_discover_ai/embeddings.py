# med_discover_ai/embeddings.py
import numpy as np
import torch
from med_discover_ai.config import (
    USE_GPU, ARTICLE_ENCODER_MODEL, QUERY_ENCODER_MODEL, EMBEDDING_MODEL,
    MAX_ARTICLE_LENGTH, MAX_QUERY_LENGTH, DEVICE, OPENAI_API_KEY
)
from openai import OpenAI, APIKeyMissingError

# --- Global Variables for Models (initialized conditionally) ---
article_tokenizer = None
article_model = None
query_tokenizer = None
query_model = None
openai_client = None

# --- Initialization ---
def initialize_models():
    """Initializes models based on GPU availability."""
    global article_tokenizer, article_model, query_tokenizer, query_model, openai_client

    if USE_GPU:
        try:
            from transformers import AutoTokenizer, AutoModel
            print("Loading MedCPT models for GPU...")
            article_tokenizer = AutoTokenizer.from_pretrained(ARTICLE_ENCODER_MODEL)
            article_model = AutoModel.from_pretrained(ARTICLE_ENCODER_MODEL).to(DEVICE)
            article_model.eval() # Set model to evaluation mode

            query_tokenizer = AutoTokenizer.from_pretrained(QUERY_ENCODER_MODEL)
            query_model = AutoModel.from_pretrained(QUERY_ENCODER_MODEL).to(DEVICE)
            query_model.eval() # Set model to evaluation mode
            print("MedCPT models loaded successfully.")
        except ImportError:
            print("Error: 'transformers' library not found. Cannot use MedCPT models.")
            # Fallback or raise error? For now, just print.
        except Exception as e:
            print(f"Error loading MedCPT models: {e}")

    # Initialize OpenAI client regardless of GPU, as it's needed for LLM generation
    # and potentially for CPU-based embeddings.
    try:
        # The client automatically uses the OPENAI_API_KEY environment variable if set.
        # If not set, it might raise an error later when an API call is made,
        # unless the key is provided dynamically (e.g., via the UI).
        openai_client = OpenAI()
        # You could add a check here if needed, but typically the error occurs on the first API call.
        # try:
        #     openai_client.models.list() # Example lightweight call to check connectivity/key
        #     print("OpenAI client initialized successfully.")
        # except APIKeyMissingError:
        #     print("Warning: OpenAI API Key is missing. Please set it via environment variable or UI.")
        # except Exception as e:
        #     print(f"Warning: Could not verify OpenAI connection: {e}")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")

# Call initialization when the module is loaded
initialize_models()

# --- Embedding Functions ---

def embed_documents(doc_chunks, batch_size=8):
    """
    Generate embeddings for document chunks using either MedCPT (GPU) or OpenAI (CPU).

    Parameters:
        doc_chunks (list): List of text chunks.
        batch_size (int): Batch size for processing (relevant for GPU).

    Returns:
        np.array: Array of embeddings. Returns empty array on failure.
    """
    if USE_GPU and article_model and article_tokenizer:
        # --- GPU Mode: Use MedCPT Article Encoder ---
        all_embeds = []
        print(f"Embedding {len(doc_chunks)} chunks using MedCPT (GPU)...")
        for i in range(0, len(doc_chunks), batch_size):
            batch = doc_chunks[i:i + batch_size]
            try:
                with torch.no_grad(): # Disable gradient calculations for inference
                    # Tokenize the batch
                    encoded = article_tokenizer(
                        batch,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                        max_length=MAX_ARTICLE_LENGTH
                    )
                    # Move tensors to the configured device (GPU)
                    encoded = {key: val.to(DEVICE) for key, val in encoded.items()}

                    # Get model outputs
                    outputs = article_model(**encoded)
                    # Extract the [CLS] token embedding (or mean pooling if preferred)
                    batch_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_embeds.append(batch_embeds)
            except Exception as e:
                print(f"Error embedding batch {i//batch_size}: {e}")
                # Optionally decide how to handle errors, e.g., skip batch, return partial results

        if all_embeds:
            print("Document embedding finished.")
            return np.vstack(all_embeds)
        else:
            print("Warning: No document embeddings were generated.")
            return np.array([])
    elif not USE_GPU and openai_client and EMBEDDING_MODEL:
        # --- CPU Mode: Use OpenAI Embedding API ---
        embeddings = []
        print(f"Embedding {len(doc_chunks)} chunks using OpenAI '{EMBEDDING_MODEL}' (CPU)...")
        for i, text in enumerate(doc_chunks):
            try:
                # Ensure text is not empty or just whitespace
                if not text or text.isspace():
                    print(f"Warning: Skipping empty chunk at index {i}.")
                    # Add a zero vector or handle as appropriate
                    # For now, skipping, which might cause index misalignment if not handled later
                    continue

                response = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
                embed = response.data[0].embedding
                embeddings.append(embed)
            except APIKeyMissingError:
                 print("Error: OpenAI API Key is missing. Cannot generate embeddings. Please set the key.")
                 return np.array([]) # Stop embedding process
            except Exception as e:
                print(f"Error embedding chunk {i} with OpenAI: {e}")
                # Handle error, e.g., skip chunk, add zero vector
        print("Document embedding finished.")
        return np.array(embeddings)
    else:
        # Handle cases where models/clients aren't initialized
        if USE_GPU:
            print("Error: MedCPT models not available for GPU embedding.")
        else:
            print(f"Error: OpenAI client or model '{EMBEDDING_MODEL}' not available for CPU embedding.")
        return np.array([])

def embed_query(query):
    """
    Generate embedding for a single query using either MedCPT (GPU) or OpenAI (CPU).

    Parameters:
        query (str): Input query.

    Returns:
        np.array: Query embedding (shape [1, embedding_dim]). Returns None on failure.
    """
    if USE_GPU and query_model and query_tokenizer:
        # --- GPU Mode: Use MedCPT Query Encoder ---
        print("Embedding query using MedCPT (GPU)...")
        try:
            with torch.no_grad():
                encoded = query_tokenizer(
                    query,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=MAX_QUERY_LENGTH
                )
                encoded = {key: val.to(DEVICE) for key, val in encoded.items()}
                outputs = query_model(**encoded)
                query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            print("Query embedding finished.")
            return query_embedding # Shape should be [1, 768]
        except Exception as e:
            print(f"Error embedding query with MedCPT: {e}")
            return None
    elif not USE_GPU and openai_client and EMBEDDING_MODEL:
        # --- CPU Mode: Use OpenAI Embedding API ---
        print(f"Embedding query using OpenAI '{EMBEDDING_MODEL}' (CPU)...")
        try:
             # Ensure query is not empty
            if not query or query.isspace():
                print("Error: Cannot embed empty query.")
                return None

            response = openai_client.embeddings.create(input=query, model=EMBEDDING_MODEL)
            embed = response.data[0].embedding
            print("Query embedding finished.")
            return np.array([embed]) # Shape needs to be [1, 1536] for FAISS search
        except APIKeyMissingError:
            print("Error: OpenAI API Key is missing. Cannot generate query embedding. Please set the key.")
            return None
        except Exception as e:
            print(f"Error embedding query with OpenAI: {e}")
            return None
    else:
        # Handle cases where models/clients aren't initialized
        if USE_GPU:
            print("Error: MedCPT query model not available for GPU embedding.")
        else:
            print(f"Error: OpenAI client or model '{EMBEDDING_MODEL}' not available for CPU embedding.")
        return None

