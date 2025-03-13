def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks.
    
    Parameters:
        text (str): Input text.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words.
        
    Returns:
        list: List of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        start += (chunk_size - overlap)
    return chunks
