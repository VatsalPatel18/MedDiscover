# med_discover_ai/gradio_app.py
import gradio as gr
import os
import json
import signal
import threading
import time
import traceback
import faiss # Import faiss here if used directly (e.g., in placeholder functions)
import ollama # Import ollama library for checking models

# Import necessary functions and config values
from med_discover_ai.pdf_utils import extract_text_from_pdf
from med_discover_ai.chunking import chunk_text
from med_discover_ai.embeddings import embed_documents, initialize_models as initialize_embedding_models
from med_discover_ai.index import build_faiss_index, save_index, load_index
from med_discover_ai.retrieval import search_and_rerank, load_metadata, initialize_reranker
# Import the ollama_client instance from llm_inference
from med_discover_ai.llm_inference import get_llm_answer, initialize_llm_clients, ollama_client
from med_discover_ai.config import (
    INDEX_SAVE_PATH, DOC_META_PATH, OPENAI_API_KEY, OLLAMA_BASE_URL,
    AVAILABLE_EMBEDDING_MODELS, AVAILABLE_LLM_MODELS,
    DEFAULT_EMBEDDING_MODEL_NAME, DEFAULT_LLM_MODEL,
    DEFAULT_MAX_TOKENS, DEFAULT_K, DEFAULT_RERANK_ENABLED,
    USE_GPU, DEVICE,
    get_embedding_model_id, get_embedding_dimension # Import helpers
)
import openai # Needed for API key setting re-initialization

# --- Global State ---
global_index = None
global_metadata = None
index_ready = False
current_index_embedding_dim = None # Store dimension of loaded index

# --- Initialization ---
# (initialize_backend function remains the same as previous version)
def initialize_backend():
    """Initialize models and load existing index/metadata if available."""
    global global_index, global_metadata, index_ready, current_index_embedding_dim
    print("--- Initializing Backend ---")
    initialize_embedding_models() # Loads MedCPT if GPU, initializes OpenAI client
    initialize_reranker()         # Loads MedCPT cross-encoder if GPU
    initialize_llm_clients()      # Initializes OpenAI and Ollama clients

    # Try to load existing index and metadata
    if os.path.exists(INDEX_SAVE_PATH) and os.path.exists(DOC_META_PATH):
        print("Attempting to load existing index and metadata...")
        expected_dim_on_load = get_embedding_dimension(DEFAULT_EMBEDDING_MODEL_NAME) # Get dim for default model
        print(f"Attempting load with expected dimension: {expected_dim_on_load} (based on default: {DEFAULT_EMBEDDING_MODEL_NAME})")
        loaded_index = load_index(INDEX_SAVE_PATH, expected_dim=expected_dim_on_load)
        loaded_meta = load_metadata(DOC_META_PATH) # Use load_metadata from retrieval.py

        if loaded_index is not None and loaded_meta is not None:
            global_index = loaded_index
            global_metadata = loaded_meta
            current_index_embedding_dim = global_index.d # Store the actual dimension
            index_ready = True
            print(f"Existing index (dim={current_index_embedding_dim}) and metadata loaded successfully.")
            default_model_dim = get_embedding_dimension(DEFAULT_EMBEDDING_MODEL_NAME)
            if current_index_embedding_dim != default_model_dim:
                 print(f"Warning: Loaded index dimension ({current_index_embedding_dim}) does not match the default embedding model ('{DEFAULT_EMBEDDING_MODEL_NAME}' dim={default_model_dim}). Ensure the correct embedding model is selected in the UI.")
        else:
            if loaded_index is None: print("Failed to load existing index (check dimension mismatch or file corruption).")
            if loaded_meta is None: print("Failed to load existing metadata.")
            print("Please process PDFs.")
            index_ready = False
    else:
        print("No existing index/metadata found. Please process PDFs.")
        index_ready = False
    print("--- Backend Initialization Complete ---")

initialize_backend()

# --- Gradio Interface Functions ---

# (set_api_key function remains the same)
def set_api_key(api_key):
    """Sets the OpenAI API key and re-initializes relevant clients."""
    status_message = "Please enter a valid OpenAI API key."
    if api_key and api_key != "YOUR_OPENAI_API_KEY_HERE" and not api_key.isspace():
        try:
            print("Setting OpenAI API Key...")
            os.environ["OPENAI_API_KEY"] = api_key
            initialize_embedding_models() # Re-initializes OpenAI client in embeddings
            initialize_llm_clients()      # Re-initializes OpenAI client in llm_inference
            status_message = "API key set successfully! Clients re-initialized."
            print(status_message)
        except Exception as e:
            status_message = f"Error setting API key: {e}"
            print(status_message)
    return status_message


# (process_pdfs_interface function remains the same)
def process_pdfs_interface(pdf_files_list, selected_embedding_model, progress=gr.Progress()):
    """
    Processes uploaded PDFs using the *selected* embedding model.
    """
    global global_index, global_metadata, index_ready, current_index_embedding_dim
    index_ready = False # Reset status

    if not pdf_files_list: return "No PDF files uploaded."
    if not selected_embedding_model: return "Error: No embedding model selected."

    model_id = get_embedding_model_id(selected_embedding_model)
    expected_dimension = get_embedding_dimension(selected_embedding_model)
    if not model_id: return f"Error: Invalid embedding model selected: {selected_embedding_model}"

    use_ip = (model_id == "ncbi/MedCPT-Article-Encoder")
    if use_ip and not USE_GPU: return f"Error: Cannot process PDFs with '{selected_embedding_model}' (MedCPT) without a GPU."

    print(f"Processing {len(pdf_files_list)} PDF files using: {selected_embedding_model} (Dim: {expected_dimension})")
    # ... (Rest of PDF processing logic: extract, chunk, embed, build index, save) ...
    # (Code omitted for brevity - same as previous version)
    all_chunks = []
    metadata_list = []
    doc_id_counter = 0
    total_files = len(pdf_files_list)
    progress(0, desc="Starting PDF Processing...")
    try:
        for i, file_obj in enumerate(pdf_files_list):
            file_path = file_obj.name
            original_filename = os.path.basename(file_path)
            progress((i + 0.1) / total_files, desc=f"Extracting text from {original_filename}...")
            text = extract_text_from_pdf(file_path)
            if not text or text.startswith("Error reading"): continue
            progress((i + 0.3) / total_files, desc=f"Chunking text for {original_filename}...")
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            if not chunks: continue
            for chunk_id, chunk_text_content in enumerate(chunks):
                metadata_list.append({"doc_id": doc_id_counter, "filename": original_filename, "chunk_id": chunk_id, "text": chunk_text_content})
                all_chunks.append(chunk_text_content)
            doc_id_counter += 1
        if not all_chunks: return "Error: No text could be extracted or chunked."
        progress(0.8, desc=f"Embedding {len(all_chunks)} chunks...")
        embeddings = embed_documents(all_chunks, selected_embedding_model)
        if embeddings is None or embeddings.shape[0] == 0: return f"Error: Failed to generate embeddings using {selected_embedding_model}."
        if embeddings.shape[0] != len(metadata_list): return "Error: Mismatch between processed chunks and generated embeddings."
        if embeddings.shape[1] != expected_dimension: return f"Error: Generated embedding dimension ({embeddings.shape[1]}) != expected ({expected_dimension})."
        progress(0.9, desc="Building FAISS index...")
        index = build_faiss_index(embeddings, use_ip_metric=use_ip)
        if index is None: return "Error: Failed to build FAISS index."
        progress(0.95, desc="Saving index and metadata...")
        index_saved = save_index(index, INDEX_SAVE_PATH)
        meta_saved = False
        try:
            os.makedirs(os.path.dirname(DOC_META_PATH), exist_ok=True)
            with open(DOC_META_PATH, "w", encoding='utf-8') as f: json.dump(metadata_list, f, indent=4)
            meta_saved = True
        except Exception as e: print(f"Error saving metadata: {e}")
        if index_saved and meta_saved:
            global_index = index
            global_metadata = metadata_list
            current_index_embedding_dim = index.d
            index_ready = True
            progress(1.0, desc="Processing Complete!")
            return f"Processed {doc_id_counter} PDFs. Index built ({index.ntotal} vectors, Dim: {index.d}) using '{selected_embedding_model}'. Ready!"
        else:
            if not index_saved and os.path.exists(INDEX_SAVE_PATH): os.remove(INDEX_SAVE_PATH)
            if not meta_saved and os.path.exists(DOC_META_PATH): os.remove(DOC_META_PATH)
            return "Error: Index or metadata saving failed."
    except Exception as e:
        print(f"Error during PDF processing: {e}\n{traceback.format_exc()}")
        index_ready = False
        return f"An error occurred: {e}."


# (query_chat_interface function remains the same)
def query_chat_interface(query, selected_embedding_model, selected_llm, k_value, rerank_enabled, max_tokens_value):
    """
    Handles user queries using selected models and parameters.
    """
    global global_index, global_metadata, index_ready, current_index_embedding_dim
    default_error_msg = "An error occurred during query processing. Please check console logs."
    no_info_msg = "Could not find relevant information for your query in the processed documents."

    if not index_ready or global_index is None or global_metadata is None: return "Error: Index is not ready.", "Please process PDF files first."
    if not query or query.isspace(): return "Please enter a query.", ""
    if not selected_embedding_model: return "Error: No embedding model selected.", "Select embedding model in Setup."
    if not selected_llm: return "Error: No LLM selected.", "Select response LLM in Setup."

    try: k_value = int(k_value)
    except: k_value = DEFAULT_K; print(f"Warning: Invalid k value, using default {k_value}")
    try: max_tokens_value = int(max_tokens_value)
    except: max_tokens_value = DEFAULT_MAX_TOKENS; print(f"Warning: Invalid max_tokens value, using default {max_tokens_value}")

    expected_dimension = get_embedding_dimension(selected_embedding_model)
    if expected_dimension is None: return f"Error: Cannot determine dimension for '{selected_embedding_model}'.", "Config error."
    # Check index dimension vs selected model dimension
    if current_index_embedding_dim is None: # Check if index is loaded at all
         return "Error: Index dimension not available. Process PDFs first.", "Index not loaded."
    if expected_dimension != current_index_embedding_dim:
        error_msg = (f"Error: Dimension mismatch! Index dim={current_index_embedding_dim}, "
                     f"Selected model ('{selected_embedding_model}') dim={expected_dimension}. \n\n"
                     f"Solution: Re-process PDFs with '{selected_embedding_model}' OR select the correct model used for the current index.")
        return error_msg, error_msg

    print(f"\n--- Processing Query ---")
    print(f"Query: '{query}'")
    print(f"Embedding Model: {selected_embedding_model} (Dim: {expected_dimension})")
    print(f"LLM: {selected_llm}")
    print(f"Retrieve K: {k_value}, Re-rank: {rerank_enabled}, Max Tokens: {max_tokens_value}")
    print(f"------------------------")

    try:
        print("Step 1: Search & Re-rank...")
        candidates = search_and_rerank(query=query, index=global_index, doc_metadata=global_metadata,
                                       embedding_model_display_name=selected_embedding_model, k=k_value, enable_rerank=rerank_enabled)
        if not candidates: print("Step 1 Result: No candidates found."); return no_info_msg, no_info_msg
        print(f"Step 1 Result: Retrieved {len(candidates)} candidates.")
        top_cand = candidates[0]; ret_score = top_cand.get('retrieval_score', 'N/A'); rerank_score = top_cand.get('rerank_score', 'N/A')
        print(f"Top candidate: File='{top_cand.get('filename', 'N/A')}', Chunk={top_cand.get('chunk_id', 'N/A')}, RetScore={ret_score:.4f}, RerankScore={rerank_score if isinstance(rerank_score, str) else f'{rerank_score:.4f}'}")

        print(f"Step 2: Generate LLM answer ({selected_llm})...")
        answer, context_text = get_llm_answer(query, candidates, llm_model=selected_llm, max_tokens=max_tokens_value)
        if answer.startswith("Error:"):
             print(f"Step 2 Result: LLM generation failed. Error: {answer}")
             context_display = f"Context from {len(candidates)} chunks was prepared but LLM failed:\n\n" + context_text
             return answer, context_display
        else:
            print(f"Step 2 Result: Answer generated.")

        context_display = f"Context generated from {len(candidates)} retrieved chunks (scroll for more):\n\n" + context_text
        return answer, context_display
    except Exception as e:
        print(f"Error during query processing: {e}\n{traceback.format_exc()}")
        return default_error_msg, f"Error details: {e}"


# --- Updated Function: Check Ollama Model Availability ---
def check_ollama_model_availability(selected_llm_model):
    """Checks if the selected Ollama model is available locally."""
    if not selected_llm_model or not selected_llm_model.startswith("ollama:"):
        return "N/A (Select an Ollama model first)"

    # Ensure the ollama_client is available (initialized in llm_inference)
    if ollama_client is None:
        # Try to re-initialize, maybe the server started after the app did
        initialize_llm_clients()
        if ollama_client is None: # Check again
             return f"Error: Ollama client not connected (Check server at {OLLAMA_BASE_URL})"

    ollama_model_tag = selected_llm_model.split(':', 1)[1] # e.g., gemma2:2b

    try:
        print(f"Checking availability for Ollama model: {ollama_model_tag}...")
        # Use ollama.list() which is the correct method in the library
        response = ollama.list()
        local_models_data = response.get('models', []) # Get the list of models, default to empty list

        # Add robust checking for the structure
        is_available = False
        for model_info in local_models_data:
            if isinstance(model_info, dict) and 'name' in model_info:
                if model_info['name'] == ollama_model_tag:
                    is_available = True
                    break
            else:
                print(f"Warning: Unexpected item format in ollama list response: {model_info}")


        if is_available:
            print(f"Model {ollama_model_tag} is available locally.")
            return f"✅ '{ollama_model_tag}' is available locally."
        else:
            print(f"Model {ollama_model_tag} not found locally.")
            pull_command = f"ollama pull {ollama_model_tag}"
            # Check if base model exists (e.g., if 'llama3:8b' requested, check if 'llama3' exists)
            base_model_name = ollama_model_tag.split(':')[0]
            base_available = any(
                m.get('name', '').startswith(base_model_name + ':')
                for m in local_models_data if isinstance(m, dict)
            )
            if base_available:
                 return f"❌ '{ollama_model_tag}' not found locally (but other '{base_model_name}' versions might exist).\nRun in terminal: `{pull_command}`"
            else:
                 return f"❌ '{ollama_model_tag}' not found locally.\nRun in terminal: `{pull_command}`"

    except Exception as e:
        # Catch potential connection errors or other issues with ollama.list()
        print(f"Error checking Ollama models via API: {e}")
        return f"Error checking Ollama ({type(e).__name__}). Is server running at {OLLAMA_BASE_URL}?"


# (shutdown_app function remains the same)
def shutdown_app():
    """Attempts to gracefully shut down the Gradio server."""
    print("Shutdown requested...")
    def stop():
        time.sleep(1)
        try: print("Sending SIGTERM..."); os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e: print(f"Error sending SIGTERM: {e}. Attempting os._exit..."); os._exit(1)
    threading.Thread(target=stop, daemon=True).start()
    return "Server shutdown initiated..."


# --- Build Gradio Interface ---
def build_interface():
    """Creates the Gradio interface layout and connects components."""
    with gr.Blocks(theme=gr.themes.Soft(), title="MedDiscover v1.2") as demo:
        gr.Markdown("# 🩺 MedDiscover: Biomedical Research Assistant (v1.2)")
        gr.Markdown("Upload PDFs, select models, ask questions, get RAG-powered answers.")

        with gr.Row():
            # --- Left Column (Setup & Controls) ---
            with gr.Column(scale=2):
                gr.Markdown("### Setup & Controls")

                # API Key (Remains at top level)
                with gr.Group():
                    gr.Markdown("**1. OpenAI API Key (Optional)**")
                    gr.Markdown(f"_Needed if using OpenAI models. Ollama runs locally (ensure server at {OLLAMA_BASE_URL} is active)._")
                    api_key_input = gr.Textbox(label="API Key", type="password", placeholder="Enter your sk-... key here",
                                               value=OPENAI_API_KEY if OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE" else "")
                    api_key_button = gr.Button("Set/Update API Key")
                    api_key_status = gr.Textbox(label="API Key Status", interactive=False, lines=1)

                # PDF Processing (Remains at top level)
                with gr.Group():
                    gr.Markdown("**2. Process PDFs**")
                    gr.Markdown("_Processing builds a search index using the selected **Embedding Model** (chosen in Advanced Settings below). Takes time._")
                    pdf_input = gr.File(label="Upload PDF Files", file_count="multiple", file_types=[".pdf"], type="filepath")
                    process_button = gr.Button("Process Uploaded PDFs", variant="primary")
                    process_output = gr.Textbox(label="Processing Status", interactive=False, lines=3)


                # Advanced Settings Accordion (MOVED MODEL SELECTIONS HERE)
                with gr.Accordion("Advanced Settings & Model Selection", open=False):
                     # GPU Status Indicator
                     with gr.Group():
                          gr.Markdown("**Hardware Status**")
                          gpu_status_text = f"GPU Detected: {'Yes' if USE_GPU else 'No'} ({DEVICE.upper()})"
                          gpu_status = gr.Textbox(label="Acceleration", value=gpu_status_text, interactive=False)

                     # Embedding Model Dropdown
                     with gr.Group():
                          gr.Markdown("**Embedding Model Selection**")
                          gr.Markdown("_Must match the model used when processing PDFs._")
                          embedding_model_dropdown = gr.Dropdown(
                              label="Embedding Model", choices=list(AVAILABLE_EMBEDDING_MODELS.keys()),
                              value=DEFAULT_EMBEDDING_MODEL_NAME,
                              info="Model used to create vector embeddings.", interactive=True
                          )

                     # LLM Selection Dropdown & Check Button
                     with gr.Group():
                          gr.Markdown("**Response LLM Selection**")
                          with gr.Row():
                               llm_model_dropdown = gr.Dropdown(
                                   label="Response LLM", choices=AVAILABLE_LLM_MODELS,
                                   value=DEFAULT_LLM_MODEL,
                                   info="Model used to generate the final answer.", interactive=True, scale=3
                               )
                               check_ollama_button = gr.Button("Check Ollama Model", scale=1)
                          # Increased lines for status output
                          ollama_status_output = gr.Textbox(label="Ollama Model Status", interactive=False, lines=3)


                     # Retrieval Settings (Remain here)
                     with gr.Group():
                          gr.Markdown("**Retrieval & Generation Parameters**")
                          k_slider = gr.Slider(label="Chunks to Retrieve (k)", minimum=1, maximum=20, step=1, value=DEFAULT_K)
                          rerank_checkbox = gr.Checkbox(label="Enable Re-ranking", value=DEFAULT_RERANK_ENABLED,
                                                        info=f"Use MedCPT Cross-Encoder. Requires GPU ({'Available' if USE_GPU else 'Not Available'}).",
                                                        interactive=USE_GPU)
                          max_tokens_slider = gr.Slider(label="Max Output Tokens", minimum=20, maximum=1000, step=10, value=DEFAULT_MAX_TOKENS)


                # Server Control (Remains at bottom)
                with gr.Group():
                    gr.Markdown("**Server Control**")
                    shutdown_button = gr.Button("Shutdown Server")
                    shutdown_output = gr.Textbox(label="Server Status", interactive=False, lines=1)

            # --- Right Column (Chat Interface - No Changes Here) ---
            with gr.Column(scale=3):
                gr.Markdown("### Chat Interface")
                gr.Markdown("_Enter query below. Ensure the **Embedding Model** selected in Advanced Settings matches the one used to process PDFs._")
                query_input = gr.Textbox(label="Enter your query here", lines=3, placeholder="e.g., What biomarkers are associated with Gaucher Disease?")
                chat_button = gr.Button("Get Answer", variant="primary")
                gr.Markdown("---")
                gr.Markdown("**Generated Answer**")
                answer_output = gr.Textbox(label="Answer", lines=8, interactive=False, show_copy_button=True)
                gr.Markdown("**Retrieved Context Used**")
                context_output = gr.Textbox(label="Context", lines=15, interactive=False, show_copy_button=True)


        # --- Connect Components ---
        # Setup Column Connections
        api_key_button.click(fn=set_api_key, inputs=api_key_input, outputs=api_key_status)
        process_button.click(
            fn=process_pdfs_interface,
            inputs=[pdf_input, embedding_model_dropdown],
            outputs=process_output,
            show_progress="full"
        )
        # Ollama Check Button Connection
        check_ollama_button.click(
            fn=check_ollama_model_availability,
            inputs=llm_model_dropdown, # Pass the selected LLM
            outputs=ollama_status_output # Output to the status box
        )
        shutdown_button.click(fn=shutdown_app, inputs=None, outputs=shutdown_output, api_name="shutdown")

        # Chat Column Connections (Inputs now come from Accordion)
        chat_inputs = [
            query_input,
            embedding_model_dropdown, # From Accordion
            llm_model_dropdown,       # From Accordion
            k_slider,                 # From Accordion
            rerank_checkbox,          # From Accordion
            max_tokens_slider         # From Accordion
        ]
        chat_outputs = [answer_output, context_output]

        chat_button.click(fn=query_chat_interface, inputs=chat_inputs, outputs=chat_outputs, api_name="query")
        query_input.submit(fn=query_chat_interface, inputs=chat_inputs, outputs=chat_outputs)

    return demo

# --- Main Execution ---
if __name__ == "__main__":
    try: from med_discover_ai.index import load_index, build_faiss_index
    except ImportError: print("Warning: Could not import functions from index.py.")

    med_discover_app = build_interface()
    print("Launching MedDiscover Gradio App...")
    med_discover_app.launch(server_name="0.0.0.0", server_port=7860)

