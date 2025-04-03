# med_discover_ai/llm_inference.py
import openai # Import base library
# Removed direct import of APIKeyMissingError
from med_discover_ai.config import DEFAULT_LLM_MODEL # Import default

# --- Global OpenAI Client ---
# Attempt to initialize the client. It might fail if the key is missing initially.
try:
    # Use openai.OpenAI() which automatically looks for the API key
    # in environment variables or other configurations.
    client = openai.OpenAI()
    # You could add an optional check here, but errors are usually caught during API calls.
    # client.models.list()
except Exception as e:
    # This initialization itself shouldn't fail due to missing key,
    # but other issues might occur.
    print(f"LLM Inference Warning: Could not initialize OpenAI client: {e}")
    client = None

# --- LLM Answer Generation ---
def get_llm_answer(query, retrieved_candidates, llm_model=DEFAULT_LLM_MODEL):
    """
    Generate an answer using a specified OpenAI LLM based on retrieved candidate texts.

    Parameters:
        query (str): The user's original query.
        retrieved_candidates (list): A list of candidate dictionaries, sorted by relevance.
                                     Each dictionary should have a "text" key.
        llm_model (str): The specific OpenAI model to use (e.g., "gpt-4o", "gpt-4o-mini-2024-07-18").

    Returns:
        tuple: (str, str) containing:
               - The generated answer string, or an error message.
               - The context string used for generation, or an empty string if no context.
    """
    global client # Access the potentially initialized client

    # Check if client is initialized, try again if not (e.g., key set via UI)
    if client is None:
        print("LLM client not ready, attempting re-initialization...")
        try:
            client = openai.OpenAI()
            # Optional check: client.models.list()
            print("LLM client re-initialized.")
        except Exception as e:
             # If re-initialization fails, return error immediately
             return f"Error: Failed to initialize OpenAI client: {e}", ""

    # Proceed with context preparation and API call
    if not retrieved_candidates:
        print("Warning: No candidates provided to generate LLM answer. Querying without context.")
        context_text = ""
        prompt = f"""
        Answer the following question concisely, in as few words as possible, based on general knowledge.

        Question: {query}

        Answer (in minimal words):
        """
    else:
        context_text = " ".join([cand["text"] for cand in retrieved_candidates])
        prompt = f"""
        You are Med-Discover, an assistant for enhancing disease discovery, using provided context from research papers.
        Use ONLY the context below to answer the question in as few words as possible. If the context doesn't contain the answer, say "Information not found in context".

        Context:
        {context_text}
        ---
        Question: {query}

        Answer (in minimal words, based ONLY on context):
        """

    print(f"Generating LLM answer using model: {llm_model}...")

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1
        )
        answer = response.choices[0].message.content.strip()
        print("LLM answer generated successfully.")
        return answer, context_text

    # Catch specific OpenAI errors using the openai.ErrorName convention
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
        # Catch any other unexpected errors during the API call
        print(f"Error during LLM inference with model {llm_model}: {e}")
        return f"Error generating answer: {e}", context_text

