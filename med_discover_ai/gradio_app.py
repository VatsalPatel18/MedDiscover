import gradio as gr
from med_discover_ai.pdf_utils import extract_text_from_pdf
from med_discover_ai.chunking import chunk_text
from med_discover_ai.embeddings import embed_documents
from med_discover_ai.index import build_faiss_index, save_index
from med_discover_ai.retrieval import search_with_rerank
from med_discover_ai.llm_inference import get_llm_answer
import os
import json
import signal
import threading
import time

# Global variables to store the built index and metadata during the session.
global_index = None
global_metadata = None

def process_pdfs(pdf_paths):
    """
    Process uploaded PDF files: extract text, chunk the text, compute embeddings,
    and build the FAISS index.
    """
    import os, json
    from med_discover_ai.pdf_utils import extract_text_from_pdf
    from med_discover_ai.chunking import chunk_text
    from med_discover_ai.embeddings import embed_documents
    from med_discover_ai.index import build_faiss_index

    TEMP_PDF_FOLDER = "./temp_pdfs"
    if not os.path.exists(TEMP_PDF_FOLDER):
        os.makedirs(TEMP_PDF_FOLDER, exist_ok=True)

    all_chunks = []
    metadata = []
    doc_id = 0

    # Gradio now provides `pdf_paths` as a list of file paths (strings).
    for file_path in pdf_paths:
        original_filename = os.path.basename(file_path)

        # Extract text from the PDF
        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            metadata.append({
                "doc_id": doc_id,
                "filename": original_filename,
                "chunk_id": i,
                "text": chunk
            })
            all_chunks.append(chunk)

        doc_id += 1

    # Compute embeddings and build the FAISS index
    embeddings = embed_documents(all_chunks)
    print("Embeddings shape:", embeddings.shape)
    index = build_faiss_index(embeddings)
    save_index(index, "faiss_index.bin")

    # Store index & metadata in global variables
    global global_index, global_metadata
    global_index = index
    global_metadata = metadata

    # Optionally save metadata to disk
    with open("doc_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return "PDFs processed and index built successfully!"

def query_chat(query):
    """
    Process a query by retrieving candidate chunks and generating an LLM answer.
    """
    if global_index is None or global_metadata is None:
        return "Please process PDFs first."
    candidates = search_with_rerank(query, global_index, global_metadata, k=5)
    answer, context_text = get_llm_answer(query, candidates)
    return f"Answer: {answer}\n\nContext (first 300 chars): {context_text[:300]}... with actual context length being : {len(context_text)}"

def set_api_key(api_key):
    import os
    os.environ["OPENAI_API_KEY"] = api_key
    return "API key set successfully!"

def shutdown_app():
    def stop():
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM) 
    threading.Thread(target=stop).start()
    return "Server shutting down..."

def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Med-Discover")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Setup")

                # 1) API Key
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="Enter your API key here"
                )
                api_key_button = gr.Button("Set API Key")
                api_key_status = gr.Textbox(label="API Key Status", interactive=False)

                # 2) PDF Upload & Process
                pdf_input = gr.File(label="Upload PDF files", file_count="multiple",type="filepath")
                process_button = gr.Button("Process PDFs")
                process_output = gr.Textbox(label="Processing Status")

                # 3) Shutdown
                shutdown_button = gr.Button("Shutdown")
                shutdown_output = gr.Textbox(label="Server Status")

            with gr.Column(scale=1):
                gr.Markdown("### Chat")
                query_input = gr.Textbox(label="Enter your query")
                chat_button = gr.Button("Get Answer")
                chat_output = gr.Textbox(label="Response", lines=10)

        # --- HOOK UP EVENTS ---
        # Set API key
        api_key_button.click(
            fn=set_api_key,
            inputs=api_key_input,
            outputs=api_key_status
        )

        # Process PDFs
        process_button.click(
            fn=process_pdfs,
            inputs=pdf_input,
            outputs=process_output
        )

        # Chat
        chat_button.click(
            fn=query_chat,
            inputs=query_input,
            outputs=chat_output
        )

        # Shutdown
        shutdown_button.click(
            fn=shutdown_app,
            outputs=shutdown_output
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
