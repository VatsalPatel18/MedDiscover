# med_discover_ai/llm_inference.py
import openai # Import base library
from med_discover_ai.config import DEFAULT_LLM_MODEL, DEFAULT_MAX_TOKENS # Import defaults

# --- Global OpenAI Client ---
# Attempt to initialize the client. It might fail if the key is missing initially.
try:
    client = openai.OpenAI()
except Exception as e:
    print(f"LLM Inference Warning: Could not initialize OpenAI client: {e}")
    client = None

# --- LLM Answer Generation ---
def get_llm_answer(query, retrieved_candidates, llm_model=DEFAULT_LLM_MODEL, max_tokens=DEFAULT_MAX_TOKENS):
    """
    Generate an answer using a specified OpenAI LLM based on retrieved candidate texts.

    Parameters:
        query (str): The user's original query.
        retrieved_candidates (list): A list of candidate dictionaries, sorted by relevance.
                                     Each dictionary should have a "text" key.
        llm_model (str): The specific OpenAI model to use.
        max_tokens (int): The maximum number of tokens to generate in the response.

    Returns:
        tuple: (str, str) containing:
               - The generated answer string, or an error message.
               - The full context string used for generation, or an empty string if no context.
    """
    global client # Access the potentially initialized client

    # Check if client is initialized, try again if not (e.g., key set via UI)
    if client is None:
        print("LLM client not ready, attempting re-initialization...")
        try:
            client = openai.OpenAI()
            print("LLM client re-initialized.")
        except Exception as e:
             return f"Error: Failed to initialize OpenAI client: {e}", ""

    # Prepare context and prompt
    if not retrieved_candidates:
        print("Warning: No candidates provided to generate LLM answer. Querying without context.")
        context_text = "" # No context available
        prompt = f"""
        Answer the following question concisely, based on general knowledge.

        Question: {query}

        Answer:
        """
    else:
        # Join the text of all retrieved candidates to form the context
        context_text = "\n\n---\n\n".join([f"Source: {cand.get('filename', 'N/A')} | Chunk: {cand.get('chunk_id', 'N/A')}\n\n{cand['text']}" for cand in retrieved_candidates])

        prompt = f"""
        You are Med-Discover, an assistant for enhancing disease discovery, using provided context from research papers.
        Use ONLY the context below to answer the question concisely. If the context doesn't contain the answer, state that the information was not found in the provided documents.

        Context:
        --- START CONTEXT ---
        {context_text}
        --- END CONTEXT ---

        Question: {query}

        Concise Answer (based ONLY on context):
        """

    print(f"Generating LLM answer using model: {llm_model}, max_tokens: {max_tokens}...")

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, # Use the max_tokens value passed from UI/config
            temperature=0.1
        )
        answer = response.choices[0].message.content.strip()
        print("LLM answer generated successfully.")
        # Return the generated answer and the full context used
        return answer, context_text

    # Catch specific OpenAI errors
    except openai.APIKeyMissingError:
        print("Error: OpenAI API Key is missing. Cannot generate LLM answer.")
        return "Error: OpenAI API Key is missing. Please set it via environment variable or the UI.", context_text
    except openai.RateLimitError:
        print("Error: OpenAI rate limit exceeded.")
        return "Error: Rate limit exceeded. Please try again later.", context_text
    except openai.AuthenticationError:
         print("Error: OpenAI authentication failed. Check your API key.")
         return "Error: Invalid OpenAI API Key.", context_text
    except Exception as e:
        print(f"Error during LLM inference with model {llm_model}: {e}")
        return f"Error generating answer: {e}", context_text

