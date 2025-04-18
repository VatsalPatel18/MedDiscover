# med_discover_ai/gradio_app.py
import gradio as gr
import os
import json
import signal
import threading
import time
import traceback
import faiss
import ollama # Import ollama library
import subprocess # Import subprocess module
import re # Import regex for sanitization
from datetime import datetime
import pandas as pd

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
    get_embedding_model_id, get_embedding_dimension
)
import openai

# --- Global State ---
global_index = None
global_metadata = None
index_ready = False
current_index_embedding_dim = None
 
# --- Logging Setup ---
LOG_DIR = "./logs"
JSON_LOG_FILE = os.path.join(LOG_DIR, "logs.jsonl")
CSV_LOG_FILE = os.path.join(LOG_DIR, "logs.csv")
os.makedirs(LOG_DIR, exist_ok=True)

def log_query(entry):
    try:
        with open(JSON_LOG_FILE, "a", encoding="utf-8") as f:
            json.dump(entry, f)
            f.write("\n")
    except Exception as e:
        print(f"Error logging query: {e}")

def download_logs():
    try:
        entries = []
        with open(JSON_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))
        if not entries:
            return None
        rows = []
        for e in entries:
            row = {
                "timestamp": e.get("timestamp"),
                "query": e.get("query"),
                "embedding_model": e.get("embedding_model"),
                "llm_model": e.get("llm_model"),
                "answer": e.get("answer"),
                "retrieved_context": " ||| ".join(e.get("retrieved_context", []))
            }
            usage = e.get("usage", {})
            row["prompt_tokens"] = usage.get("prompt_tokens")
            row["completion_tokens"] = usage.get("completion_tokens")
            row["total_tokens"] = usage.get("total_tokens")
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(CSV_LOG_FILE, index=False)
        return CSV_LOG_FILE
    except Exception as e:
        print(f"Error preparing logs for download: {e}")
        return None

# --- Initialization ---
# (initialize_backend function remains the same)
def initialize_backend():
    """Initialize models and load existing index/metadata if available."""
    global global_index, global_metadata, index_ready, current_index_embedding_dim
    print("--- Initializing Backend ---")
    initialize_embedding_models()
    initialize_reranker()
    initialize_llm_clients()
    if os.path.exists(INDEX_SAVE_PATH) and os.path.exists(DOC_META_PATH):
        print("Attempting to load existing index and metadata...")
        expected_dim_on_load = get_embedding_dimension(DEFAULT_EMBEDDING_MODEL_NAME)
        print(f"Attempting load with expected dimension: {expected_dim_on_load} (based on default: {DEFAULT_EMBEDDING_MODEL_NAME})")
        loaded_index = load_index(INDEX_SAVE_PATH, expected_dim=expected_dim_on_load)
        loaded_meta = load_metadata(DOC_META_PATH)
        if loaded_index is not None and loaded_meta is not None:
            global_index = loaded_index
            global_metadata = loaded_meta
            current_index_embedding_dim = global_index.d
            index_ready = True
            print(f"Existing index (dim={current_index_embedding_dim}) and metadata loaded successfully.")
            default_model_dim = get_embedding_dimension(DEFAULT_EMBEDDING_MODEL_NAME)
            if current_index_embedding_dim != default_model_dim:
                 print(f"Warning: Loaded index dimension ({current_index_embedding_dim}) != default embedding model ('{DEFAULT_EMBEDDING_MODEL_NAME}' dim={default_model_dim}).")
        else:
            if loaded_index is None: print("Failed to load existing index.")
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
    status_message = "Please enter a valid OpenAI API key."
    if api_key and api_key != "YOUR_OPENAI_API_KEY_HERE" and not api_key.isspace():
        try:
            print("Setting OpenAI API Key...")
            os.environ["OPENAI_API_KEY"] = api_key
            initialize_embedding_models()
            initialize_llm_clients()
            status_message = "API key set successfully! Clients re-initialized."
            print(status_message)
        except Exception as e:
            status_message = f"Error setting API key: {e}"
            print(status_message)
    return status_message

# (process_pdfs_interface function remains the same)
def process_pdfs_interface(pdf_files_list, selected_embedding_model, progress=gr.Progress()):
    global global_index, global_metadata, index_ready, current_index_embedding_dim
    index_ready = False
    if not pdf_files_list: return "No PDF files uploaded."
    if not selected_embedding_model: return "Error: No embedding model selected."
    model_id = get_embedding_model_id(selected_embedding_model)
    expected_dimension = get_embedding_dimension(selected_embedding_model)
    if not model_id: return f"Error: Invalid embedding model: {selected_embedding_model}"
    use_ip = (model_id == "ncbi/MedCPT-Article-Encoder")
    if use_ip and not USE_GPU: return f"Error: Cannot use '{selected_embedding_model}' without GPU."
    print(f"Processing {len(pdf_files_list)} PDFs using: {selected_embedding_model} (Dim: {expected_dimension})")
    all_chunks, metadata_list, doc_id_counter = [], [], 0
    total_files = len(pdf_files_list)
    progress(0, desc="Starting PDF Processing...")
    try:
        # ... (PDF processing logic - extract, chunk) ...
        for i, file_obj in enumerate(pdf_files_list):
             file_path = file_obj.name; original_filename = os.path.basename(file_path)
             progress((i + 0.1) / total_files, desc=f"Extracting: {original_filename}...")
             text = extract_text_from_pdf(file_path)
             if not text or text.startswith("Error reading"): continue
             progress((i + 0.3) / total_files, desc=f"Chunking: {original_filename}...")
             chunks = chunk_text(text, chunk_size=500, overlap=50)
             if not chunks: continue
             for chunk_id, chunk_text_content in enumerate(chunks):
                 metadata_list.append({"doc_id": doc_id_counter, "filename": original_filename, "chunk_id": chunk_id, "text": chunk_text_content})
                 all_chunks.append(chunk_text_content)
             doc_id_counter += 1
        if not all_chunks: return "Error: No text extracted/chunked."
        progress(0.8, desc=f"Embedding {len(all_chunks)} chunks...")
        embeddings = embed_documents(all_chunks, selected_embedding_model)
        if embeddings is None or embeddings.shape[0] == 0: return f"Error generating embeddings with {selected_embedding_model}."
        if embeddings.shape[0] != len(metadata_list): return "Error: Mismatch between chunks and embeddings."
        if embeddings.shape[1] != expected_dimension: return f"Error: Embedding dim ({embeddings.shape[1]}) != expected ({expected_dimension})."
        progress(0.9, desc="Building FAISS index...")
        index = build_faiss_index(embeddings, use_ip_metric=use_ip)
        if index is None: return "Error: Failed to build FAISS index."
        progress(0.95, desc="Saving index & metadata...")
        index_saved = save_index(index, INDEX_SAVE_PATH)
        meta_saved = False
        try:
            os.makedirs(os.path.dirname(DOC_META_PATH), exist_ok=True)
            with open(DOC_META_PATH, "w", encoding='utf-8') as f: json.dump(metadata_list, f, indent=4)
            meta_saved = True
        except Exception as e: print(f"Error saving metadata: {e}")
        if index_saved and meta_saved:
            global_index, global_metadata, current_index_embedding_dim, index_ready = index, metadata_list, index.d, True
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
    global global_index, global_metadata, index_ready, current_index_embedding_dim
    default_error_msg = "Error during query processing. Check logs."
    no_info_msg = "Could not find relevant info."
    if not index_ready or global_index is None or global_metadata is None: return "Error: Index not ready.", "Process PDFs first."
    if not query or query.isspace(): return "Enter a query.", ""
    if not selected_embedding_model: return "Error: Select embedding model.", "Select embedding model."
    if not selected_llm: return "Error: Select LLM.", "Select LLM."
    try: k_value = int(k_value)
    except: k_value = DEFAULT_K; print(f"Warning: Invalid k, using {k_value}")
    try: max_tokens_value = int(max_tokens_value)
    except: max_tokens_value = DEFAULT_MAX_TOKENS; print(f"Warning: Invalid max_tokens, using {max_tokens_value}")
    expected_dimension = get_embedding_dimension(selected_embedding_model)
    if expected_dimension is None: return f"Error: Cannot get dimension for '{selected_embedding_model}'.", "Config error."
    if current_index_embedding_dim is None: return "Error: Index dimension unavailable.", "Index not loaded."
    if expected_dimension != current_index_embedding_dim:
        error_msg = (f"Error: Dim mismatch! Index={current_index_embedding_dim}, "
                     f"Selected Model ('{selected_embedding_model}')={expected_dimension}. \n\n"
                     f"Solution: Re-process PDFs with '{selected_embedding_model}' OR select the correct model.")
        return error_msg, error_msg
    print(f"\n--- Processing Query ---")
    print(f"Query: '{query}'\nEmbedding: {selected_embedding_model} (Dim: {expected_dimension})\nLLM: {selected_llm}\nK: {k_value}, Re-rank: {rerank_enabled}, MaxTokens: {max_tokens_value}\n------------------------")
    try:
        print("Step 1: Search & Re-rank...")
        candidates = search_and_rerank(query=query, index=global_index, doc_metadata=global_metadata,
                                       embedding_model_display_name=selected_embedding_model, k=k_value, enable_rerank=rerank_enabled)
        if not candidates: print("Step 1 Result: No candidates."); return no_info_msg, no_info_msg
        print(f"Step 1 Result: Retrieved {len(candidates)} candidates.")
        top_cand = candidates[0]; ret_score = top_cand.get('retrieval_score', 'N/A'); rerank_score = top_cand.get('rerank_score', 'N/A')
        print(f"Top candidate: File='{top_cand.get('filename', 'N/A')}', Chunk={top_cand.get('chunk_id', 'N/A')}, RetScore={ret_score:.4f}, RerankScore={rerank_score if isinstance(rerank_score, str) else f'{rerank_score:.4f}'}")
        print(f"Step 2: Generate LLM answer ({selected_llm})...")
        # Call LLM and capture usage metrics
        answer, context_text, usage = get_llm_answer(query, candidates, llm_model=selected_llm, max_tokens=max_tokens_value)
        if answer.startswith("Error:"):
             print(f"Step 2 Result: LLM failed. Error: {answer}")
             context_display = f"Context prepared ({len(candidates)} chunks) but LLM failed.\n(Check Ollama status/pull model if needed)"
             # Log failure case
             log_entry = {
                 "timestamp": datetime.now().isoformat(),
                 "query": query,
                 "embedding_model": selected_embedding_model,
                 "llm_model": selected_llm,
                 "k": k_value,
                 "rerank_enabled": rerank_enabled,
                 "max_tokens": max_tokens_value,
                 "retrieved_context": [c.get("text", "") for c in candidates],
                 "answer": answer,
                 "usage": usage
             }
             log_query(log_entry)
             return answer, context_display
        else:
            print(f"Step 2 Result: Answer generated.")
        context_display = f"Context from {len(candidates)} chunks (scroll):\n\n" + context_text
        # Log successful case
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "embedding_model": selected_embedding_model,
            "llm_model": selected_llm,
            "k": k_value,
            "rerank_enabled": rerank_enabled,
            "max_tokens": max_tokens_value,
            "retrieved_context": [c.get("text", "") for c in candidates],
            "answer": answer,
            "usage": usage
        }
        log_query(log_entry)
        return answer, context_display
    except Exception as e:
        print(f"Error during query processing: {e}\n{traceback.format_exc()}")
        return default_error_msg, f"Error details: {e}"

# (ensure_ollama_model_available function remains the same)
def ensure_ollama_model_available(selected_llm_model):
    """Checks if the selected Ollama model is available locally, and if not, attempts to pull it."""
    if not selected_llm_model or not selected_llm_model.startswith("ollama:"):
        return "N/A (Select an Ollama model first)"
    if ollama_client is None:
        initialize_llm_clients()
        if ollama_client is None:
             return f"Error: Ollama client not connected (Check server at {OLLAMA_BASE_URL})"
    ollama_model_tag = selected_llm_model.split(':', 1)[1]
    if not re.match(r"^[a-zA-Z0-9:_\-\.]+$", ollama_model_tag):
        print(f"Error: Invalid Ollama model tag format: {ollama_model_tag}")
        return f"Error: Invalid model tag format."
    try:
        print(f"Checking if Ollama model exists: {ollama_model_tag}...")
        # Correct Client.show signature: takes model name as positional arg
        ollama_client.show(ollama_model_tag)
        print(f"Model {ollama_model_tag} verified successfully.")
        return f"✅ '{ollama_model_tag}' is available locally."
    except ollama.ResponseError as e:
        # Model not found locally: attempt to pull via Ollama client
        if getattr(e, 'status_code', None) == 404 or "not found" in str(e).lower():
            print(f"Model {ollama_model_tag} not found locally. Pulling via Ollama client...")
            # Use Ollama client pull API if available
            if ollama_client and hasattr(ollama_client, 'pull'):
                try:
                    response = ollama_client.pull(ollama_model_tag, stream=False)
                    # If streaming, consume generator
                    if hasattr(response, '__iter__'):
                        for _ in response:
                            pass
                    # Verify now available
                    ollama_client.show(ollama_model_tag)
                    print(f"Model {ollama_model_tag} pulled successfully.")
                    return f"✅ '{ollama_model_tag}' pulled and available locally."
                except Exception as pull_err:
                    print(f"Error pulling Ollama model {ollama_model_tag} via client: {pull_err}")
                    # Fallback to CLI pull
            # Fallback: shell out to 'ollama pull'
            try:
                command = ["ollama", "pull", ollama_model_tag]
                print(f"Running command: {' '.join(command)}")
                completed = subprocess.run(command, capture_output=True, text=True, check=True)
                print(f"CLI pull stdout: {completed.stdout}")
                # Verify model after CLI pull
                ollama_client.show(ollama_model_tag)
                return f"✅ '{ollama_model_tag}' pulled via CLI and available locally."
            except FileNotFoundError:
                print("Error: 'ollama' command not found. Is Ollama installed and in the system PATH?")
                return "Error: 'ollama' command not found in PATH."
            except subprocess.CalledProcessError as cpe:
                print(f"Error during CLI pull of Ollama model {ollama_model_tag}: {cpe.stderr}")
                return f"Error pulling model via CLI: {cpe.stderr.strip()}"
            except Exception as pull_err:
                print(f"Unexpected error pulling Ollama model {ollama_model_tag}: {pull_err}")
                return f"Error pulling model: {pull_err}"
        else:
            print(f"Error checking Ollama model {ollama_model_tag} with ollama.show(): {e} (Status: {getattr(e,'status_code', 'N/A')})")
            return f"Error checking Ollama ({getattr(e,'status_code', '')}): {e}"
    except Exception as check_err:
        print(f"Error checking Ollama models: {check_err}")
        if "connection refused" in str(check_err).lower():
             return f"Error: Cannot connect to Ollama at {OLLAMA_BASE_URL}. Is the server running?"
        return f"Error checking Ollama: {check_err}"

# (shutdown_app function remains the same)
def shutdown_app():
    print("Shutdown requested...")
    def stop():
        time.sleep(1)
        try: print("Sending SIGTERM..."); os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e: print(f"Error sending SIGTERM: {e}. Forcing exit..."); os._exit(1)
    threading.Thread(target=stop, daemon=True).start()
    return "Server shutdown initiated..."

# --- Build Gradio Interface ---
def build_interface():
    """Creates the Gradio interface layout and connects components."""
    with gr.Blocks(theme=gr.themes.Soft(), title="MedDiscover v1.3") as demo:
        gr.Markdown("# 🩺 MedDiscover: Biomedical Research Assistant (v1.3)")
        gr.Markdown("Upload PDFs, select models (Ollama models pulled automatically if missing), ask questions.")

        with gr.Row():
            # --- Left Column (Setup & Controls) ---
            with gr.Column(scale=2):
                gr.Markdown("### Setup & Controls")
                # API Key
                with gr.Group():
                    gr.Markdown("**1. OpenAI API Key (Optional)**")
                    gr.Markdown(f"_Needed for OpenAI models. Ollama runs locally (check {OLLAMA_BASE_URL})._")
                    api_key_input = gr.Textbox(label="API Key", type="password", placeholder="sk-...", value=OPENAI_API_KEY if OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE" else "")
                    api_key_button = gr.Button("Set/Update API Key")
                    api_key_status = gr.Textbox(label="API Key Status", interactive=False, lines=1)
                # PDF Processing
                with gr.Group():
                    gr.Markdown("**2. Process PDFs**")
                    gr.Markdown("_Builds search index using **Embedding Model** chosen below._")
                    pdf_input = gr.File(label="Upload PDF Files", file_count="multiple", file_types=[".pdf"], type="filepath")
                    process_button = gr.Button("Process Uploaded PDFs", variant="primary")
                    process_output = gr.Textbox(label="Processing Status", interactive=False, lines=3)
                # Advanced Settings Accordion
                with gr.Accordion("Advanced Settings & Model Selection", open=False):
                     with gr.Group(): # Group Hardware + Embedding
                          gr.Markdown("**Hardware & Embedding**")
                          gpu_status = gr.Textbox(label="Hardware Acceleration", value=f"GPU: {'Yes' if USE_GPU else 'No'} ({DEVICE.upper()})", interactive=False)
                          embedding_model_dropdown = gr.Dropdown(label="Embedding Model", choices=list(AVAILABLE_EMBEDDING_MODELS.keys()), value=DEFAULT_EMBEDDING_MODEL_NAME, info="Must match model used for PDF processing.", interactive=True)
                     with gr.Group(): # Group LLM + Check/Pull Button
                          gr.Markdown("**Response LLM**")
                          with gr.Row():
                               llm_model_dropdown = gr.Dropdown(label="LLM", choices=AVAILABLE_LLM_MODELS, value=DEFAULT_LLM_MODEL, info="Generates the final answer.", interactive=True, scale=3)
                               # *** REMOVED tooltip argument from the line below ***
                               ensure_ollama_button = gr.Button("Ensure Ollama Model", scale=1)
                          ollama_status_output = gr.Textbox(label="Ollama Model Status", interactive=False, lines=3, placeholder="Click 'Ensure Ollama Model' after selecting an Ollama model")
                     with gr.Group(): # Group Retrieval Params
                          gr.Markdown("**Retrieval & Generation Parameters**")
                          k_slider = gr.Slider(label="Chunks (k)", minimum=1, maximum=20, step=1, value=DEFAULT_K)
                          rerank_checkbox = gr.Checkbox(label="Re-rank", value=DEFAULT_RERANK_ENABLED, info=f"MedCPT Cross-Encoder. Needs GPU ({'Available' if USE_GPU else 'Not Available'}).", interactive=USE_GPU)
                          max_tokens_slider = gr.Slider(label="Max Tokens", minimum=20, maximum=1000, step=10, value=DEFAULT_MAX_TOKENS)
                # Server Control
                with gr.Group():
                    gr.Markdown("**Server Control**")
                    shutdown_button = gr.Button("Shutdown Server")
                    shutdown_output = gr.Textbox(label="Server Status", interactive=False, lines=1)
                    # Logs download
                    gr.Markdown("**Logs**")
                    download_logs_button = gr.Button("Download Logs")
                    download_logs_file = gr.File(label="Logs File", interactive=False)

            # --- Right Column (Chat Interface) ---
            with gr.Column(scale=3):
                gr.Markdown("### Chat Interface")
                gr.Markdown("_Enter query. Ensure **Embedding Model** in Advanced Settings matches index._")
                query_input = gr.Textbox(label="Enter query", lines=3, placeholder="e.g., Biomarkers for Gaucher Disease?")
                chat_button = gr.Button("Get Answer", variant="primary")
                gr.Markdown("---")
                gr.Markdown("**Generated Answer**")
                answer_output = gr.Textbox(label="Answer", lines=8, interactive=False, show_copy_button=True)
                gr.Markdown("**Retrieved Context Used**")
                context_output = gr.Textbox(label="Context", lines=15, interactive=False, show_copy_button=True)

        # --- Connect Components ---
        api_key_button.click(fn=set_api_key, inputs=api_key_input, outputs=api_key_status)
        process_button.click(fn=process_pdfs_interface, inputs=[pdf_input, embedding_model_dropdown], outputs=process_output, show_progress="full")
        # Connect the Ensure button
        ensure_ollama_button.click(
            fn=ensure_ollama_model_available,
            inputs=llm_model_dropdown,
            outputs=ollama_status_output
        )
        shutdown_button.click(fn=shutdown_app, inputs=None, outputs=shutdown_output, api_name="shutdown")
        # Connect Logs Download
        download_logs_button.click(fn=download_logs, inputs=None, outputs=download_logs_file)
        chat_inputs = [query_input, embedding_model_dropdown, llm_model_dropdown, k_slider, rerank_checkbox, max_tokens_slider]
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

