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
    AVAILABLE_LLM_MODELS, DEFAULT_LLM_MODEL,
    DEFAULT_K, DEFAULT_RERANK_ENABLED, USE_GPU
)

# --- Global State ---
# These hold the active index and metadata for the current session
global_index = None
global_metadata = None
# Flag to indicate if the index is ready
index_ready = False

# --- Initialization ---
def initialize_backend():
    """Initialize models and load existing index/metadata if available."""
    global global_index, global_metadata, index_ready
    print("Initializing backend...")
    initialize_embedding_models() # Load embedding models (MedCPT or prepares OpenAI client)
    initialize_reranker()       # Load re-ranker model (MedCPT cross-encoder if GPU)

    # Try to load existing index and metadata
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

# Call initialization when the Gradio app starts
initialize_backend()

# --- Gradio Interface Functions ---

def set_api_key(api_key):
    """Sets the OpenAI API key in the environment."""
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE" or api_key.isspace():
        return "Please enter a valid OpenAI API key."
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        # Re-initialize models/clients that depend on the key
        initialize_embedding_models() # Re-initializes OpenAI client
        # LLM client might also need re-initialization if it failed before
        # (Assuming get_llm_answer handles client re-check)
        print("OpenAI API Key set in environment.")
        return "API key set successfully! Backend re-initialized."
    except Exception as e:
        print(f"Error setting API key: {e}")
        return f"Error setting API key: {e}"

def process_pdfs_interface(pdf_files_list, progress=gr.Progress()):
    """
    Gradio interface function to process uploaded PDFs.
    Handles text extraction, chunking, embedding, index building, and saving.

    Parameters:
        pdf_files_list (list): List of file paths provided by gr.File component.
        progress (gr.Progress): Gradio progress tracker.

    Returns:
        str: Status message indicating success or failure.
    """
    global global_index, global_metadata, index_ready
    index_ready = False # Reset status

    if not pdf_files_list:
        return "No PDF files uploaded. Please upload files to process."

    print(f"Processing {len(pdf_files_list)} PDF files...")
    all_chunks = []
    metadata = []
    doc_id_counter = 0

    # Setup progress tracking
    total_files = len(pdf_files_list)
    progress(0, desc="Starting PDF Processing...")

    try:
        for i, file_obj in enumerate(pdf_files_list):
            file_path = file_obj.name # Get the actual file path from the Gradio File object
            original_filename = os.path.basename(file_path)
            progress((i + 0.1) / total_files, desc=f"Extracting text from {original_filename}...")
            print(f"Processing file: {original_filename} ({file_path})")

            # 1. Extract Text
            text = extract_text_from_pdf(file_path)
            if not text or text.startswith("Error reading"):
                print(f"Warning: Could not extract text from {original_filename}. Skipping.")
                continue # Skip this file

            # 2. Chunk Text
            progress((i + 0.3) / total_files, desc=f"Chunking text for {original_filename}...")
            chunks = chunk_text(text, chunk_size=500, overlap=50) # Using config values ideally
            if not chunks:
                 print(f"Warning: No chunks generated for {original_filename}. Skipping.")
                 continue

            # 3. Prepare Metadata and Collect Chunks
            for chunk_id, chunk_text_content in enumerate(chunks):
                metadata.append({
                    "doc_id": doc_id_counter,
                    "filename": original_filename,
                    "chunk_id": chunk_id,
                    "text": chunk_text_content
                })
                all_chunks.append(chunk_text_content)

            doc_id_counter += 1
            print(f"Finished initial processing for {original_filename}.")

        if not all_chunks:
            return "Error: No text could be extracted or chunked from the provided PDFs."

        # 4. Embed all collected chunks
        progress(0.8, desc=f"Embedding {len(all_chunks)} text chunks...")
        print(f"Starting embedding process for {len(all_chunks)} chunks...")
        embeddings = embed_documents(all_chunks)
        if embeddings is None or embeddings.shape[0] == 0:
             return "Error: Failed to generate embeddings for the text chunks."
        if embeddings.shape[0] != len(all_chunks):
            # This indicates some chunks might have been skipped during embedding
            print(f"Warning: Number of embeddings ({embeddings.shape[0]}) does not match number of chunks ({len(all_chunks)}). Metadata might be misaligned.")
            # Adjust metadata to match embeddings - this is complex, safer to error out for now
            return "Error: Mismatch between chunks and generated embeddings. Please check logs."


        print(f"Embeddings generated successfully. Shape: {embeddings.shape}")

        # 5. Build FAISS Index
        progress(0.9, desc="Building FAISS index...")
        print("Building FAISS index...")
        index = build_faiss_index(embeddings)
        if index is None:
            return "Error: Failed to build the FAISS index."

        # 6. Save Index and Metadata & Update Global State
        progress(0.95, desc="Saving index and metadata...")
        print("Saving index and metadata...")
        index_saved = save_index(index, INDEX_SAVE_PATH)
        meta_saved = False
        try:
            # Ensure directory exists for metadata
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
        print(traceback.format_exc()) # Print detailed traceback
        index_ready = False # Ensure index is marked as not ready
        return f"An error occurred: {e}. Check console logs for details."


def query_chat_interface(query, llm_model, k_value, rerank_enabled):
    """
    Gradio interface function to handle user queries, perform retrieval,
    re-ranking (optional), and LLM generation.

    Parameters:
        query (str): The user's query text.
        llm_model (str): The selected LLM model name.
        k_value (int): The number of chunks to retrieve (k).
        rerank_enabled (bool): Whether to enable re-ranking.

    Returns:
        str: The formatted response string including the answer and context info.
    """
    global global_index, global_metadata, index_ready

    if not index_ready or global_index is None or global_metadata is None:
        return "Error: Index is not ready. Please process PDF files first."

    if not query or query.isspace():
        return "Please enter a query."

    print(f"Received query: '{query}' with LLM={llm_model}, k={k_value}, re-rank={rerank_enabled}")

    try:
        # 1. Search and Re-rank
        print("Performing search and re-rank...")
        # Ensure k_value is an integer
        try:
            k_value = int(k_value)
        except ValueError:
            print(f"Warning: Invalid k value '{k_value}', using default {DEFAULT_K}.")
            k_value = DEFAULT_K

        candidates = search_and_rerank(
            query=query,
            index=global_index,
            doc_metadata=global_metadata,
            k=k_value,
            enable_rerank=rerank_enabled
        )

        if not candidates:
            return "Could not find relevant information for your query in the processed documents."

        print(f"Retrieved {len(candidates)} candidates after processing.")
        # Display top candidate info for debugging/info
        top_cand = candidates[0]
        ret_score = top_cand.get('retrieval_score', 'N/A')
        rerank_score = top_cand.get('rerank_score', 'N/A')
        print(f"Top candidate: File='{top_cand.get('filename', 'N/A')}', Chunk={top_cand.get('chunk_id', 'N/A')}, RetScore={ret_score}, RerankScore={rerank_score}")


        # 2. Generate LLM Answer
        print("Generating LLM answer...")
        answer, context_text = get_llm_answer(query, candidates, llm_model=llm_model)

        # 3. Format Response
        response = f"**Answer:**\n{answer}\n\n---\n"
        response += f"**Context Used (from {len(candidates)} chunks, top source shown):**\n"
        response += f"Source: {top_cand.get('filename', 'N/A')} (Chunk {top_cand.get('chunk_id', 'N/A')})\n"
        # Show a snippet of the context used
        context_snippet = context_text[:300].replace('\n', ' ') # Limit length and remove newlines for display
        response += f"Snippet: {context_snippet}...\n"
        response += f"(Retrieval Score: {ret_score:.4f}"
        if 'rerank_score' in top_cand:
             response += f", Re-rank Score: {rerank_score:.4f})"
        else:
             response += ")"


        return response

    except Exception as e:
        print(f"An unexpected error occurred during query processing: {e}")
        print(traceback.format_exc()) # Print detailed traceback
        return f"An error occurred: {e}. Check console logs for details."


def shutdown_app():
    """Attempts to gracefully shut down the Gradio server."""
    print("Shutdown requested...")
    def stop():
        time.sleep(1) # Give Gradio a moment to process the request
        try:
            os.kill(os.getpid(), signal.SIGTERM) # Send termination signal
        except Exception as e:
            print(f"Error sending SIGTERM: {e}")
            # Fallback for environments where SIGTERM might not work as expected
            os._exit(1)
    # Run shutdown in a separate thread to allow the Gradio response to be sent
    threading.Thread(target=stop).start()
    return "Server shutdown initiated. You may need to close the window/tab manually."

# --- Build Gradio Interface ---
def build_interface():
    """Creates the Gradio interface layout and connects components."""
    with gr.Blocks(theme=gr.themes.Soft(), title="MedDiscover") as demo:
        gr.Markdown("# 🩺 MedDiscover: Biomedical Research Assistant")
        gr.Markdown("Upload research papers (PDF), ask questions, and get answers powered by RAG and LLMs.")

        with gr.Tabs():
            # --- Setup Tab ---
            with gr.TabItem("Setup & PDF Processing"):
                gr.Markdown("### 1. Configure OpenAI API Key")
                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="Enter your sk-... key here",
                        value=OPENAI_API_KEY if OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE" else "",
                        scale=3
                    )
                    api_key_button = gr.Button("Set API Key", scale=1)
                api_key_status = gr.Textbox(label="API Key Status", interactive=False)

                gr.Markdown("### 2. Upload and Process PDFs")
                gr.Markdown("Upload the PDF documents you want to query. Processing involves text extraction, chunking, embedding generation, and index building. This may take time depending on the number and size of PDFs and your hardware (GPU significantly speeds up embedding).")
                # Changed type to 'filepath' as it seemed required by the original process_pdfs logic
                pdf_input = gr.File(
                    label="Upload PDF Files",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath" # Passes list of file paths
                )
                process_button = gr.Button("Process Uploaded PDFs", variant="primary")
                process_output = gr.Textbox(label="Processing Status", interactive=False, lines=3)

                gr.Markdown("### 3. Server Control")
                shutdown_button = gr.Button("Shutdown Server")
                shutdown_output = gr.Textbox(label="Server Status", interactive=False)


            # --- Chat Tab ---
            with gr.TabItem("Chat & Query"):
                gr.Markdown("### Ask Questions About Your Documents")
                gr.Markdown("Ensure PDFs have been processed successfully before asking questions.")

                with gr.Row():
                    with gr.Column(scale=3):
                         query_input = gr.Textbox(label="Enter your query here", lines=3, placeholder="e.g., What biomarkers are associated with Gaucher Disease?")
                         chat_button = gr.Button("Get Answer", variant="primary")
                    with gr.Column(scale=1):
                        gr.Markdown("#### Query Options")
                        llm_model_dropdown = gr.Dropdown(
                            label="LLM Model",
                            choices=AVAILABLE_LLM_MODELS,
                            value=DEFAULT_LLM_MODEL, # Use the default from config
                            info="Select the OpenAI model for answer generation."
                        )
                        # Advanced Settings Accordion
                        with gr.Accordion("Advanced Retrieval Settings", open=False):
                             k_slider = gr.Slider(
                                 label="Number of Chunks (k)",
                                 minimum=1,
                                 maximum=20,
                                 step=1,
                                 value=DEFAULT_K, # Use default from config
                                 info="How many text chunks to retrieve initially."
                             )
                             rerank_checkbox = gr.Checkbox(
                                 label="Enable Re-ranking (GPU Only)",
                                 value=DEFAULT_RERANK_ENABLED, # Use default from config
                                 info="Use MedCPT Cross-Encoder for relevance re-ranking (requires GPU)."
                             )


                chat_output = gr.Textbox(label="Response", lines=10, interactive=False)


        # --- Connect Components ---
        # Setup Tab Connections
        api_key_button.click(
            fn=set_api_key,
            inputs=api_key_input,
            outputs=api_key_status
        )
        process_button.click(
            fn=process_pdfs_interface,
            inputs=pdf_input,
            outputs=process_output
        )
        shutdown_button.click(
            fn=shutdown_app,
            inputs=None,
            outputs=shutdown_output,
            api_name="shutdown" # Optional: Define API name for programmatic access
        )

        # Chat Tab Connections
        chat_button.click(
            fn=query_chat_interface,
            inputs=[query_input, llm_model_dropdown, k_slider, rerank_checkbox],
            outputs=chat_output,
            api_name="query" # Optional: Define API name
        )
        # Allow submitting query with Enter key
        query_input.submit(
             fn=query_chat_interface,
            inputs=[query_input, llm_model_dropdown, k_slider, rerank_checkbox],
            outputs=chat_output
        )


    return demo

# --- Main Execution ---
if __name__ == "__main__":
    # Build the Gradio interface
    med_discover_app = build_interface()

    # Launch the application
    # share=True creates a public link (use with caution)
    # server_name="0.0.0.0" makes it accessible on the local network
    print("Launching MedDiscover Gradio App...")
    med_discover_app.launch(server_name="0.0.0.0", server_port=7860)
    # Add share=True if you need a public link: med_discover_app.launch(share=True)

