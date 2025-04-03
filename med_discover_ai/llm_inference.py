# med_discover_ai/llm_inference.py
import openai
from openai import OpenAI, APIKeyMissingError
from med_discover_ai.config import DEFAULT_LLM_MODEL # Import default

# --- Global OpenAI Client ---
# Reuse the client initialized in embeddings.py if possible, or initialize here.
# For simplicity here, we assume it might need its own initialization or rely on the global env var.
try:
    client = OpenAI()
    # Check if the API key is available
    # client.models.list() # Optional check - might fail if key is missing
except APIKeyMissingError:
    print("LLM Inference Warning: OpenAI API Key is missing. Set it via environment or UI.")
    client = None # Indicate client is not ready
except Exception as e:
    print(f"LLM Inference Error: Could not initialize OpenAI client: {e}")
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

    if client is None:
        # Try to re-initialize if it wasn't ready before (e.g., key set later via UI)
        try:
            client = OpenAI()
            # client.models.list() # Optional check
        except APIKeyMissingError:
             return "Error: OpenAI API Key is missing. Cannot generate answer.", ""
        except Exception as e:
             return f"Error: Failed to initialize OpenAI client: {e}", ""

    if not retrieved_candidates:
        print("Warning: No candidates provided to generate LLM answer.")
        # Decide how to handle: Query LLM without context, or return specific message?
        # For now, let's try querying without context, but adjust prompt.
        context_text = ""
        prompt = f"""
        Answer the following question concisely, in as few words as possible, based on general knowledge.

        Question: {query}

        Answer (in minimal words):
        """
    else:
        # Combine the top candidate texts into a context.
        # Consider limiting the context size if necessary, e.g., join only top 3 candidates
        context_text = " ".join([cand["text"] for cand in retrieved_candidates]) # Using all provided candidates
        # Optional: Add truncation logic here if context_text is too long for the model

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
        # Use the ChatCompletion endpoint
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                # System message sets the persona (optional but good practice)
                # {"role": "system", "content": "You are Med-Discover, an AI assistant specialized in biomedical research."},
                # User message contains the prompt with context and question
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,  # Increased slightly for potentially more nuanced short answers
            temperature=0.1 # Low temperature for factual, concise answers
            # top_p=1.0, # Default
            # frequency_penalty=0.0, # Default
            # presence_penalty=0.0 # Default
        )

        # Extract the answer from the response
        answer = response.choices[0].message.content.strip()
        print("LLM answer generated successfully.")
        return answer, context_text

    except APIKeyMissingError:
        print("Error: OpenAI API Key is missing. Cannot generate LLM answer.")
        return "Error: OpenAI API Key is missing.", context_text
    except openai.RateLimitError:
        print("Error: OpenAI rate limit exceeded.")
        return "Error: Rate limit exceeded. Please try again later.", context_text
    except openai.AuthenticationError:
         print("Error: OpenAI authentication failed. Check your API key.")
         return "Error: Invalid OpenAI API Key.", context_text
    except Exception as e:
        print(f"Error during LLM inference with model {llm_model}: {e}")
        return f"Error generating answer: {e}", context_text

