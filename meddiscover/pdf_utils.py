import os
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Parameters:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted text.
    """
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def process_pdf_folder(pdf_folder):
    """
    Process all PDF files in a folder.
    
    Parameters:
        pdf_folder (str): Path to the folder containing PDFs.
        
    Returns:
        list: List of dictionaries with filename and extracted text.
    """
    pdf_data = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(path)
            pdf_data.append({"filename": filename, "text": text})
    return pdf_data
