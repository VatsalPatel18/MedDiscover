from config import LLM_MODEL
import openai

def get_llm_answer(query, retrieved_candidates):
    """
    Generate an answer using an LLM (e.g. GPT-4) based on retrieved candidate texts.
    
    Parameters:
        query (str): The user query.
        retrieved_candidates (list): List of candidate documents.
        
    Returns:
        tuple: (answer, context_text) where answer is the LLM-generated answer.
    """
    # Combine the top candidate texts into a context.
    context_text = " ".join([cand["text"] for cand in retrieved_candidates])
    prompt = f"""
    You are a knowledgeable assistant. Use the context below to answer the question in as few words as possible.
    
    Context:
    {context_text}
    
    Question: {query}
    
    Answer (in minimal words):
    """
    # Call the OpenAI API for LLM completion.
    response = openai.Completion.create(
        engine=LLM_MODEL,
        prompt=prompt,
        max_tokens=20,
        temperature=0
    )
    answer = response.choices[0].text.strip()
    return answer, context_text
