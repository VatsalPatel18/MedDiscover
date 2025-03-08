import gradio as gr
from pdf_utils import extract_text_from_pdf
from chunking import chunk_text
from embeddings import embed_documents
from index import build_faiss_index, save_index
from retrieval import search_with_rerank
from llm_inference import get_llm_answer
import os
import json

# Global variables to store the built index and metadata during the session.
global_index = None
global_metadata = None

def process_pdfs(pdf_files):
    """
    Process uploaded PDF files: extract text, chunk the text, compute embeddings,
    and build the FAISS index.
    """
    # Save uploaded files temporarily.
    temp_folder = "./temp_pdfs"
    os.makedirs(temp_folder, exist_ok=True)
    all_chunks = []
    metadata = []
    doc_id = 0
    for file in pdf_files:
        file_path = os.path.join(temp_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            metadata.append({
                "doc_id": doc_id,
                "filename": file.name,
                "chunk_id": i,
                "text": chunk
            })
            all_chunks.append(chunk)
        doc_id += 1
    # Compute embeddings and build the FAISS index.
    embeddings = embed_documents(all_chunks)
    index = build_faiss_index(embeddings)
    # Store index and metadata globally.
    global global_index, global_metadata
    global_index = index
    global_metadata = metadata
    # Save metadata to disk (optional).
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
    return f"Answer: {answer}\n\nContext (first 300 chars): {context_text[:300]}..."

def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# RAG-LLM Metabolomics Tool")
        with gr.Tab("Setup"):
            pdf_input = gr.File(label="Upload PDF files", file_count="multiple")
            process_button = gr.Button("Process PDFs")
            process_output = gr.Textbox(label="Processing Status")
        with gr.Tab("Chat"):
            query_input = gr.Textbox(label="Enter your query")
            chat_button = gr.Button("Get Answer")
            chat_output = gr.Textbox(label="Response", lines=10)
        
        process_button.click(fn=process_pdfs, inputs=pdf_input, outputs=process_output)
        chat_button.click(fn=query_chat, inputs=query_input, outputs=chat_output)
    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
