# med_discover_ai/gradio_app.py
import gradio as gr
import os
import json
import signal
import threading
import time
import traceback # For detailed error logging

# Import necessary functions from your modules
from med_discover_ai.pdf_utils import extract_text_from_pdf
from med_discover_ai.chunking import chunk_text
from med_discover_ai.embeddings import embed_documents, initialize_models as initialize_embedding_models
from med_discover_ai.index import build_faiss_index, save_index, load_index
from med_discover_ai.retrieval import search_and_rerank, load_metadata, initialize_reranker
from med_discover_ai.llm_inference import get_llm_answer
from med_discover_ai.config import (
    INDEX_SAVE_PATH, DOC_META_PATH, OPENAI_API_KEY,
    AVAILABLE_LLM_MODELS, DEFAULT_LLM_MODEL, DEFAULT_MAX_TOKENS, # Added DEFAULT_MAX_TOKENS
    DEFAULT_K, DEFAULT_RERANK_ENABLED, USE_GPU
)

# --- Global State ---
global_index = None
global_metadata = None
index_ready = False

# --- Initialization ---
def initialize_backend():
    """Initialize models and load existing index/metadata if available."""
    global global_index, global_metadata, index_ready
    print("Initializing backend...")
    initialize_embedding_models()
    initialize_reranker()

    if os.path.exists(INDEX_SAVE_PATH) and os.path.exists(DOC_META_PATH):
        print("Attempting to load existing index and metadata...")
        loaded_index = load_index(INDEX_SAVE_PATH)
        loaded_meta = load_metadata(DOC_META_PATH)
        if loaded_index is not None and loaded_meta is not None:
            global_index = loaded_index
            global_metadata = loaded_meta
            index_ready = True
            print("Existing index and metadata loaded successfully.")
        else:
            print("Failed to load existing index or metadata. Please process PDFs.")
            index_ready = False
    else:
        print("No existing index/metadata found. Please process PDFs.")
        index_ready = False

initialize_backend()

# --- Gradio Interface Functions ---

def set_api_key(api_key):
    """Sets the OpenAI API key in the environment."""
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE" or api_key.isspace():
        return "Please enter a valid OpenAI API key."
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        initialize_embedding_models() # Re-initializes OpenAI client
        print("OpenAI API Key set in environment.")
        # Attempt to re-initialize LLM client in llm_inference module as well
        # This might require adjusting llm_inference.py if client initialization needs explicit triggering
        try:
            from med_discover_ai import llm_inference
            llm_inference.client = openai.OpenAI() # Re-assign client instance
            print("LLM Inference client re-initialized.")
        except Exception as e:
            print(f"Warning: Could not re-initialize LLM client after setting key: {e}")

        return "API key set successfully! Backend re-initialized."
    except Exception as e:
        print(f"Error setting API key: {e}")
        return f"Error setting API key: {e}"

def process_pdfs_interface(pdf_files_list, progress=gr.Progress()):
    """
    Gradio interface function to process uploaded PDFs.
    Handles text extraction, chunking, embedding, index building, and saving.
    """
    global global_index, global_metadata, index_ready
    index_ready = False # Reset status

    if not pdf_files_list:
        return "No PDF files uploaded. Please upload files to process."

    print(f"Processing {len(pdf_files_list)} PDF files...")
    all_chunks = []
    metadata = []
    doc_id_counter = 0
    total_files = len(pdf_files_list)
    progress(0, desc="Starting PDF Processing...")

    try:
        for i, file_obj in enumerate(pdf_files_list):
            file_path = file_obj.name
            original_filename = os.path.basename(file_path)
            progress((i + 0.1) / total_files, desc=f"Extracting text from {original_filename}...")
            print(f"Processing file: {original_filename} ({file_path})")
            text = extract_text_from_pdf(file_path)
            if not text or text.startswith("Error reading"):
                print(f"Warning: Could not extract text from {original_filename}. Skipping.")
                continue

            progress((i + 0.3) / total_files, desc=f"Chunking text for {original_filename}...")
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            if not chunks:
                 print(f"Warning: No chunks generated for {original_filename}. Skipping.")
                 continue

            for chunk_id, chunk_text_content in enumerate(chunks):
                metadata.append({
                    "doc_id": doc_id_counter, "filename": original_filename,
                    "chunk_id": chunk_id, "text": chunk_text_content
                })
                all_chunks.append(chunk_text_content)
            doc_id_counter += 1
            print(f"Finished initial processing for {original_filename}.")

        if not all_chunks:
            return "Error: No text could be extracted or chunked from the provided PDFs."

        progress(0.8, desc=f"Embedding {len(all_chunks)} text chunks...")
        print(f"Starting embedding process for {len(all_chunks)} chunks...")
        embeddings = embed_documents(all_chunks)
        if embeddings is None or embeddings.shape[0] == 0:
             return "Error: Failed to generate embeddings for the text chunks."
        if embeddings.shape[0] != len(metadata): # Check against metadata length now
            print(f"Warning: Number of embeddings ({embeddings.shape[0]}) does not match number of metadata entries ({len(metadata)}). This might indicate skipped chunks during embedding.")
            # Attempt to reconcile or return error
            # For now, returning error is safer
            return "Error: Mismatch between metadata entries and generated embeddings. Please check logs."

        print(f"Embeddings generated successfully. Shape: {embeddings.shape}")

        progress(0.9, desc="Building FAISS index...")
        print("Building FAISS index...")
        index = build_faiss_index(embeddings)
        if index is None:
            return "Error: Failed to build the FAISS index."

        progress(0.95, desc="Saving index and metadata...")
        print("Saving index and metadata...")
        index_saved = save_index(index, INDEX_SAVE_PATH)
        meta_saved = False
        try:
            os.makedirs(os.path.dirname(DOC_META_PATH), exist_ok=True)
            with open(DOC_META_PATH, "w", encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
            meta_saved = True
            print("Metadata saved successfully.")
        except Exception as e:
            print(f"Error saving metadata: {e}")

        if index_saved and meta_saved:
            global_index = index
            global_metadata = metadata
            index_ready = True
            progress(1.0, desc="Processing Complete!")
            return f"Successfully processed {doc_id_counter} PDFs. Index built with {index.ntotal} vectors. Ready to chat!"
        else:
            return "Error: Index or metadata saving failed. Please check logs."

    except Exception as e:
        print(f"An unexpected error occurred during PDF processing: {e}")
        print(traceback.format_exc())
        index_ready = False
        return f"An error occurred: {e}. Check console logs for details."


def query_chat_interface(query, llm_model, k_value, rerank_enabled, max_tokens_value):
    """
    Gradio interface function to handle user queries.
    Returns separate answer and context strings.

    Parameters:
        query (str): User query.
        llm_model (str): Selected LLM model.
        k_value (int): Number of chunks (k).
        rerank_enabled (bool): Re-ranking flag.
        max_tokens_value (int): Max tokens for LLM output.

    Returns:
        tuple: (str, str) containing:
               - LLM answer string (or error message).
               - Full context string (or error/status message).
    """
    global global_index, global_metadata, index_ready
    default_error_msg = "An error occurred. Please check logs."
    no_info_msg = "Could not find relevant information for your query."

    if not index_ready or global_index is None or global_metadata is None:
        return "Error: Index is not ready. Please process PDF files first.", "Please process PDFs to enable querying."

    if not query or query.isspace():
        return "Please enter a query.", "" # Return empty context if no query

    # Validate inputs
    try:
        k_value = int(k_value)
    except ValueError:
        print(f"Warning: Invalid k value '{k_value}', using default {DEFAULT_K}.")
        k_value = DEFAULT_K
    try:
        max_tokens_value = int(max_tokens_value)
    except ValueError:
        print(f"Warning: Invalid max_tokens value '{max_tokens_value}', using default {DEFAULT_MAX_TOKENS}.")
        max_tokens_value = DEFAULT_MAX_TOKENS

    print(f"Received query: '{query}' with LLM={llm_model}, k={k_value}, re-rank={rerank_enabled}, max_tokens={max_tokens_value}")

    try:
        # 1. Search and Re-rank
        print("Performing search and re-rank...")
        candidates = search_and_rerank(
            query=query, index=global_index, doc_metadata=global_metadata,
            k=k_value, enable_rerank=rerank_enabled
        )

        if not candidates:
             # Return specific message in both outputs if no candidates found
            return no_info_msg, no_info_msg

        print(f"Retrieved {len(candidates)} candidates after processing.")

        # 2. Generate LLM Answer (pass max_tokens)
        print("Generating LLM answer...")
        answer, context_text = get_llm_answer(
            query, candidates, llm_model=llm_model, max_tokens=max_tokens_value
        )

        # 3. Return separate answer and context
        # Context text is already formatted in get_llm_answer
        return answer, context_text

    except Exception as e:
        print(f"An unexpected error occurred during query processing: {e}")
        print(traceback.format_exc())
        # Return error message in both outputs
        return default_error_msg, f"Error details: {e}"

def shutdown_app():
    """Attempts to gracefully shut down the Gradio server."""
    print("Shutdown requested...")
    def stop():
        time.sleep(1)
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:
            print(f"Error sending SIGTERM: {e}")
            os._exit(1)
    threading.Thread(target=stop).start()
    return "Server shutdown initiated. You may need to close the window/tab manually."

# --- Build Gradio Interface ---
def build_interface():
    """Creates the Gradio interface layout and connects components."""
    with gr.Blocks(theme=gr.themes.Soft(), title="MedDiscover") as demo:
        gr.Markdown("# 🩺 MedDiscover: Biomedical Research Assistant")
        gr.Markdown("Upload research papers (PDF), ask questions, and get answers powered by RAG and LLMs.")

        with gr.Row():
            # --- Left Column (Setup & Controls) ---
            with gr.Column(scale=1):
                gr.Markdown("### Setup & Controls")

                # API Key
                with gr.Group():
                    gr.Markdown("**1. OpenAI API Key**")
                    api_key_input = gr.Textbox(
                        label="API Key", type="password", placeholder="Enter your sk-... key here",
                        value=OPENAI_API_KEY if OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE" else ""
                    )
                    api_key_button = gr.Button("Set API Key")
                    api_key_status = gr.Textbox(label="API Key Status", interactive=False)

                # PDF Processing
                with gr.Group():
                    gr.Markdown("**2. Process PDFs**")
                    pdf_input = gr.File(
                        label="Upload PDF Files", file_count="multiple",
                        file_types=[".pdf"], type="filepath"
                    )
                    process_button = gr.Button("Process Uploaded PDFs", variant="primary")
                    process_output = gr.Textbox(label="Processing Status", interactive=False, lines=2)

                # Query Options
                with gr.Group():
                    gr.Markdown("**3. Query Options**")
                    llm_model_dropdown = gr.Dropdown(
                        label="LLM Model", choices=AVAILABLE_LLM_MODELS,
                        value=DEFAULT_LLM_MODEL, info="Select the OpenAI model for answer generation."
                    )
                    # Advanced Settings Accordion
                    with gr.Accordion("Advanced Settings", open=False):
                         k_slider = gr.Slider(
                             label="Chunks to Retrieve (k)", minimum=1, maximum=20, step=1,
                             value=DEFAULT_K, info="How many text chunks to retrieve initially."
                         )
                         rerank_checkbox = gr.Checkbox(
                             label="Enable Re-ranking (GPU Only)", value=DEFAULT_RERANK_ENABLED,
                             info="Use MedCPT Cross-Encoder (requires GPU)."
                         )
                         max_tokens_slider = gr.Slider(
                             label="Max Output Tokens", minimum=10, maximum=500, step=5,
                             value=DEFAULT_MAX_TOKENS, info="Max tokens for the LLM's generated answer."
                         )

                # Server Control
                with gr.Group():
                    gr.Markdown("**4. Server Control**")
                    shutdown_button = gr.Button("Shutdown Server")
                    shutdown_output = gr.Textbox(label="Server Status", interactive=False)

            # --- Right Column (Chat Interface) ---
            with gr.Column(scale=3):
                gr.Markdown("### Chat Interface")
                gr.Markdown("Enter your query below after processing PDFs.")

                query_input = gr.Textbox(
                    label="Enter your query here", lines=3,
                    placeholder="e.g., What biomarkers are associated with Gaucher Disease?"
                )
                chat_button = gr.Button("Get Answer", variant="primary")

                gr.Markdown("---") # Separator

                # Answer Output
                gr.Markdown("**Generated Answer**")
                answer_output = gr.Textbox(label="Answer", lines=5, interactive=False)

                # Context Output
                gr.Markdown("**Retrieved Context Used**")
                context_output = gr.Textbox(label="Context", lines=15, interactive=False)


        # --- Connect Components ---
        # Setup Column Connections
        api_key_button.click(fn=set_api_key, inputs=api_key_input, outputs=api_key_status)
        process_button.click(fn=process_pdfs_interface, inputs=pdf_input, outputs=process_output)
        shutdown_button.click(fn=shutdown_app, inputs=None, outputs=shutdown_output, api_name="shutdown")

        # Chat Column Connections
        chat_inputs = [query_input, llm_model_dropdown, k_slider, rerank_checkbox, max_tokens_slider]
        chat_outputs = [answer_output, context_output] # Map to the two output boxes

        chat_button.click(fn=query_chat_interface, inputs=chat_inputs, outputs=chat_outputs, api_name="query")
        query_input.submit(fn=query_chat_interface, inputs=chat_inputs, outputs=chat_outputs)

    return demo

# --- Main Execution ---
if __name__ == "__main__":
    med_discover_app = build_interface()
    print("Launching MedDiscover Gradio App...")
    med_discover_app.launch(server_name="0.0.0.0", server_port=7860)

