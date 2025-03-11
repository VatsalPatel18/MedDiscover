from config import LLM_MODEL
import openai
from openai import OpenAI

def get_llm_answer(query, retrieved_candidates):
    """
    Generate an answer using an LLM based on retrieved candidate texts.
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
    client = OpenAI()

    # Use ChatCompletion instead of the old Completion
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are MedDiscover, an assitant for enhancing disease discovery. You are RAG-LLM, connected with a specific vector database."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=20,
        temperature=0
    )

    # The answer is in `message.content`
    answer = response.choices[0].message.content.strip()
    return answer, context_text

