# med_discover_ai/embeddings.py
import os
import numpy as np
import torch
import openai # Import the base library
from med_discover_ai.config import (
    USE_GPU, DEVICE,
    MAX_ARTICLE_LENGTH, MAX_QUERY_LENGTH,
    get_embedding_model_id, get_embedding_dimension # Use helper functions
)
# Import specific model names only if needed for loading logic
from med_discover_ai.config import ARTICLE_ENCODER_MODEL as MEDCPT_ARTICLE_MODEL_ID
from med_discover_ai.config import QUERY_ENCODER_MODEL as MEDCPT_QUERY_MODEL_ID
from med_discover_ai.config import OPENAI_EMBEDDING_MODEL_ID

# Allow CPU fallback for MedCPT (set ALLOW_MEDCPT_CPU=1 to force load on CPU).
ALLOW_MEDCPT_CPU = os.environ.get("ALLOW_MEDCPT_CPU", "0") == "1"

# --- Global Variables for Models (initialized conditionally) ---
# MedCPT models (loaded only if USE_GPU is True)
medcpt_article_tokenizer = None
medcpt_article_model = None
medcpt_query_tokenizer = None
medcpt_query_model = None
# OpenAI client (initialized regardless)
openai_client = None

# --- Initialization ---
def initialize_models():
    """Initializes models based on GPU availability and API keys."""
    global medcpt_article_tokenizer, medcpt_article_model, medcpt_query_tokenizer, medcpt_query_model
    global openai_client

    can_load_medcpt = USE_GPU or ALLOW_MEDCPT_CPU

    if can_load_medcpt:
        try:
            from transformers import AutoTokenizer, AutoModel
            device_note = "GPU" if USE_GPU else "CPU (ALLOW_MEDCPT_CPU=1)"
            print(f"{'GPU detected' if USE_GPU else 'GPU not available, CPU override enabled'}. Loading MedCPT models on {device_note}...")
            # Load Article Encoder
            if MEDCPT_ARTICLE_MODEL_ID:
                medcpt_article_tokenizer = AutoTokenizer.from_pretrained(MEDCPT_ARTICLE_MODEL_ID)
                medcpt_article_model = AutoModel.from_pretrained(MEDCPT_ARTICLE_MODEL_ID).to(DEVICE)
                medcpt_article_model.eval()
                print(f"MedCPT Article Encoder ({MEDCPT_ARTICLE_MODEL_ID}) loaded.")
            else:
                print("Warning: MedCPT Article Encoder model ID not configured.")

            # Load Query Encoder
            if MEDCPT_QUERY_MODEL_ID:
                medcpt_query_tokenizer = AutoTokenizer.from_pretrained(MEDCPT_QUERY_MODEL_ID)
                medcpt_query_model = AutoModel.from_pretrained(MEDCPT_QUERY_MODEL_ID).to(DEVICE)
                medcpt_query_model.eval()
                print(f"MedCPT Query Encoder ({MEDCPT_QUERY_MODEL_ID}) loaded.")
            else:
                print("Warning: MedCPT Query Encoder model ID not configured.")

        except ImportError:
            print("Error: 'transformers' library not found. Cannot use MedCPT models.")
        except Exception as e:
            print(f"Error loading MedCPT models: {e}")
    else:
        print("GPU not available and ALLOW_MEDCPT_CPU not set. MedCPT models will not be loaded.")

    # Initialize OpenAI client
    try:
        openai_client = openai.OpenAI()
        # Perform a lightweight check if API key is likely present
        try:
            _ = openai_client.models.list()  # compatibility with OpenAI SDK >=2.x
            print("OpenAI client initialized successfully.")
        except openai.AuthenticationError:
            print("Warning: OpenAI API Key is missing or invalid. OpenAI embeddings/models will fail if used.")
        except Exception as e:
            print(f"Warning: Could not verify OpenAI connection: {e}")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")

# Call initialization when the module is loaded
initialize_models()

# --- Embedding Functions ---

def embed_documents(doc_chunks, embedding_model_display_name, batch_size=8):
    """
    Generate embeddings for document chunks using the *selected* model.

    Parameters:
        doc_chunks (list): List of text chunks.
        embedding_model_display_name (str): The display name selected in the UI.
        batch_size (int): Batch size for processing (relevant for GPU/MedCPT).

    Returns:
        np.array: Array of embeddings. Returns empty array on failure.
    """
    model_id = get_embedding_model_id(embedding_model_display_name)
    print(f"Attempting to embed {len(doc_chunks)} chunks using selected model: {embedding_model_display_name} (ID: {model_id})")

    # --- Try MedCPT if selected AND available (GPU or CPU override) ---
    if (
        model_id == MEDCPT_ARTICLE_MODEL_ID
        and medcpt_article_model
        and medcpt_article_tokenizer
        and (USE_GPU or ALLOW_MEDCPT_CPU)
    ):
        all_embeds = []
        device_note = "GPU" if USE_GPU else "CPU override"
        print(f"Using MedCPT Article Encoder ({device_note}) with batch size {batch_size}...")
        for i in range(0, len(doc_chunks), batch_size):
            batch = doc_chunks[i:i + batch_size]
            try:
                with torch.no_grad():
                    encoded = medcpt_article_tokenizer(
                        batch, truncation=True, padding=True, return_tensors="pt", max_length=MAX_ARTICLE_LENGTH
                    )
                    encoded = {key: val.to(DEVICE) for key, val in encoded.items()}
                    outputs = medcpt_article_model(**encoded)
                    batch_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_embeds.append(batch_embeds)
            except Exception as e:
                print(f"Error embedding batch {i//batch_size} with MedCPT: {e}")
                # Decide handling: skip batch, add zeros, or stop? Stopping for safety.
                return np.array([])

        if all_embeds:
            print("MedCPT document embedding finished.")
            return np.vstack(all_embeds)
        else:
            print("Warning: No MedCPT document embeddings were generated.")
            return np.array([])

    # --- Try OpenAI if selected AND client available ---
    elif model_id == OPENAI_EMBEDDING_MODEL_ID and openai_client:
        embeddings = []
        print(f"Using OpenAI Embedding API ('{model_id}', CPU/Cloud)...")
        for i, text in enumerate(doc_chunks):
            try:
                if not text or text.isspace():
                    print(f"Warning: Skipping empty chunk at index {i}.")
                    # Need to handle potential index misalignment if skipping.
                    # For simplicity, let's add a zero vector of the correct dimension.
                    dim = get_embedding_dimension(embedding_model_display_name)
                    embeddings.append(np.zeros(dim, dtype=np.float32))
                    continue

                response = openai_client.embeddings.create(input=text, model=model_id)
                embed = response.data[0].embedding
                embeddings.append(embed)
            except openai.APIKeyMissingError:
                print("Error: OpenAI API Key is missing. Cannot generate embeddings. Please set the key.")
                return np.array([]) # Stop embedding process
            except openai.AuthenticationError:
                print("Error: OpenAI API Key is invalid.")
                return np.array([])
            except Exception as e:
                print(f"Error embedding chunk {i} with OpenAI: {e}")
                # Add zero vector on error to maintain alignment
                dim = get_embedding_dimension(embedding_model_display_name)
                embeddings.append(np.zeros(dim, dtype=np.float32))

        print("OpenAI document embedding finished.")
        return np.array(embeddings)

    # --- Handle Fallback/Error Cases ---
    else:
        if model_id == MEDCPT_ARTICLE_MODEL_ID:
            if not (USE_GPU or ALLOW_MEDCPT_CPU):
                print(f"Error: Cannot use MedCPT ('{embedding_model_display_name}') without GPU or ALLOW_MEDCPT_CPU=1.")
            elif not medcpt_article_model or not medcpt_article_tokenizer:
                print(f"Error: MedCPT model ('{embedding_model_display_name}') not loaded properly.")
            else:
                print(f"Error: Unknown issue preventing MedCPT usage for '{embedding_model_display_name}'.")
        elif model_id == OPENAI_EMBEDDING_MODEL_ID:
            if not openai_client:
                print(f"Error: OpenAI client not available for '{embedding_model_display_name}'. Check API key.")
            else:
                 print(f"Error: Unknown issue preventing OpenAI usage for '{embedding_model_display_name}'.")
        else:
            print(f"Error: Selected embedding model '{embedding_model_display_name}' is not recognized or supported.")

        return np.array([])


def embed_query(query, embedding_model_display_name):
    """
    Generate embedding for a single query using the *selected* model.

    Parameters:
        query (str): Input query.
        embedding_model_display_name (str): The display name selected in the UI.

    Returns:
        np.array: Query embedding (shape [1, embedding_dim]). Returns None on failure.
    """
    model_id = get_embedding_model_id(embedding_model_display_name)
    print(f"Attempting to embed query using selected model: {embedding_model_display_name} (ID: {model_id})")

    if not query or query.isspace():
        print("Error: Cannot embed empty query.")
        return None

    # --- Try MedCPT if selected AND GPU available AND model loaded ---
    if (
        model_id == MEDCPT_ARTICLE_MODEL_ID
        and medcpt_query_model
        and medcpt_query_tokenizer
        and (USE_GPU or ALLOW_MEDCPT_CPU)
    ):
        device_note = "GPU" if USE_GPU else "CPU override"
        print(f"Using MedCPT Query Encoder ({device_note})...")
        try:
            with torch.no_grad():
                encoded = medcpt_query_tokenizer(
                    query, truncation=True, padding=True, return_tensors="pt", max_length=MAX_QUERY_LENGTH
                )
                encoded = {key: val.to(DEVICE) for key, val in encoded.items()}
                outputs = medcpt_query_model(**encoded)
                query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            print("MedCPT query embedding finished.")
            return query_embedding # Shape should be [1, 768]
        except Exception as e:
            print(f"Error embedding query with MedCPT: {e}")
            return None

    # --- Try OpenAI if selected AND client available ---
    elif model_id == OPENAI_EMBEDDING_MODEL_ID and openai_client:
        print(f"Using OpenAI Embedding API ('{model_id}', CPU/Cloud)...")
        try:
            response = openai_client.embeddings.create(input=query, model=model_id)
            embed = response.data[0].embedding
            print("OpenAI query embedding finished.")
            return np.array([embed]) # Shape needs to be [1, 1536] for FAISS search
        except openai.APIKeyMissingError:
            print("Error: OpenAI API Key is missing. Cannot generate query embedding.")
            return None
        except openai.AuthenticationError:
            print("Error: OpenAI API Key is invalid.")
            return None
        except Exception as e:
            print(f"Error embedding query with OpenAI: {e}")
            return None

    # --- Handle Fallback/Error Cases ---
    else:
        if model_id == MEDCPT_ARTICLE_MODEL_ID:
             if not USE_GPU:
                 print(f"Error: Cannot use MedCPT ('{embedding_model_display_name}') for query embedding without a GPU.")
             elif not medcpt_query_model or not medcpt_query_tokenizer:
                  print(f"Error: MedCPT query model ('{embedding_model_display_name}') not loaded properly.")
             else:
                  print(f"Error: Unknown issue preventing MedCPT query embedding for '{embedding_model_display_name}'.")
        elif model_id == OPENAI_EMBEDDING_MODEL_ID:
             if not openai_client:
                 print(f"Error: OpenAI client not available for query embedding ('{embedding_model_display_name}'). Check API key.")
             else:
                  print(f"Error: Unknown issue preventing OpenAI query embedding for '{embedding_model_display_name}'.")
        else:
             print(f"Error: Selected embedding model '{embedding_model_display_name}' is not recognized or supported for query embedding.")

        return None
