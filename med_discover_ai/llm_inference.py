# med_discover_ai/llm_inference.py
import openai
import ollama # Import the ollama library
from med_discover_ai.config import (
    DEFAULT_LLM_MODEL, DEFAULT_MAX_TOKENS,
    OLLAMA_BASE_URL # Import Ollama base URL
)

# --- Global API Clients ---
openai_client = None
ollama_client = None

# --- Initialization ---
def initialize_llm_clients():
    """Initializes API clients for OpenAI and Ollama."""
    global openai_client, ollama_client

    # Initialize OpenAI Client
    try:
        openai_client = openai.OpenAI()
        # Light check (compatible with OpenAI SDK >=2.x)
        _ = openai_client.models.list()
        print("LLM Inference: OpenAI client initialized.")
    except openai.AuthenticationError:
        print("LLM Inference Warning: OpenAI API Key missing/invalid. OpenAI models unavailable.")
        openai_client = None # Ensure client is None if key is bad
    except Exception as e:
        print(f"LLM Inference Warning: Could not initialize OpenAI client: {e}")
        openai_client = None

    # Initialize Ollama Client
    try:
        ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        # Light check to see if server is reachable
        ollama_client.list() # Lists models available locally
        print(f"LLM Inference: Ollama client initialized (connected to {OLLAMA_BASE_URL}).")
    except Exception as e:
        # Catch connection errors, etc.
        print(f"LLM Inference Warning: Could not initialize Ollama client at {OLLAMA_BASE_URL}: {e}")
        print("Ensure the Ollama server is running and accessible.")
        ollama_client = None

# Initialize clients when module loads
initialize_llm_clients()

# --- LLM Answer Generation ---
def get_llm_answer(query, retrieved_candidates, llm_model=DEFAULT_LLM_MODEL, max_tokens=DEFAULT_MAX_TOKENS):
    """
    Generate an answer using the specified LLM (OpenAI or Ollama) based on retrieved context.

    Parameters:
        query (str): The user's original query.
        retrieved_candidates (list): Candidate dictionaries with 'text', 'filename', 'chunk_id'.
        llm_model (str): Identifier for the LLM (e.g., "gpt-4o", "ollama:llama3:8b").
        max_tokens (int): Maximum tokens for the generated response.

    Returns:
        tuple: (str, str) containing (answer, context_text) or (error_message, context_text).
    """
    global openai_client, ollama_client # Access clients

    # --- Prepare Context and Base Prompt ---
    if not retrieved_candidates:
        context_text = "No context provided."
        prompt_template = """
        Answer the following question concisely based on general knowledge.

        Question: {query}

        Answer:
        """
    else:
        # Format context clearly
        context_parts = []
        for i, cand in enumerate(retrieved_candidates):
             context_parts.append(
                 f"--- Context Chunk {i+1} (Source: {cand.get('filename', 'N/A')} | Chunk ID: {cand.get('chunk_id', 'N/A')}) ---\n"
                 f"{cand.get('text', 'N/A')}"
             )
        context_text = "\n\n".join(context_parts)

        prompt_template = """
        You are Med-Discover, an AI assistant specialized in biomedical research.
        Use ONLY the provided context below to answer the question concisely.
        If the context does not contain the answer, state that the information was not found in the provided documents.
        Do not add any preamble like "Based on the context...".

        --- START CONTEXT ---
        {context_text}
        --- END CONTEXT ---

        Question: {query}

        Concise Answer (based ONLY on context):
        """

    # Populate the prompt
    prompt = prompt_template.format(query=query, context_text=context_text)

    # --- Select LLM Provider and Generate ---
    answer = f"Error: Model '{llm_model}' not recognized or client unavailable." # Default error

    # Initialize usage dict for token tracking
    usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    # --- Ollama ---
    if llm_model.startswith("ollama:"):
        if ollama_client:
            ollama_model_name = llm_model.split(':', 1)[1] # Extract model name (e.g., "llama3:8b")
            print(f"Generating answer using Ollama model: {ollama_model_name} (max_tokens: {max_tokens})...")
            try:
                # Note: Ollama's `max_tokens` equivalent is often controlled via `num_predict` in options
                # Temperature is also set via options.
                response = ollama_client.chat(
                    model=ollama_model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'num_predict': max_tokens,
                        'temperature': 0.1 # Low temperature for factual answers
                    }
                )
                answer = response['message']['content'].strip()
                print("Ollama answer generated successfully.")
            except ollama.ResponseError as e:
                 print(f"Error during Ollama inference with model {ollama_model_name}: {e.error}")
                 # Check for common errors like model not found locally
                 if "model" in e.error.lower() and "not found" in e.error.lower():
                      answer = f"Error: Ollama model '{ollama_model_name}' not found locally. Pull it using 'ollama run {ollama_model_name}'."
                 else:
                      answer = f"Error during Ollama inference: {e.error}"
            except Exception as e:
                print(f"Generic error during Ollama inference with model {ollama_model_name}: {e}")
                answer = f"Error generating answer with Ollama: {e}"
        else:
            print("Error: Ollama client not available. Cannot use Ollama models.")
            answer = "Error: Ollama client not initialized. Check server connection."

    # --- OpenAI ---
    # Check if it's a known OpenAI model (or assume it is if not Ollama)
    # --- OpenAI (any non-ollama model) ---
    elif not llm_model.startswith("ollama:"):
        if openai_client:
            print(f"Generating answer using OpenAI model: {llm_model} (max_tokens: {max_tokens})...")
            try:
                response = openai_client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                answer = response.choices[0].message.content.strip()
                # Capture usage metrics if available
                if hasattr(response, "usage") and response.usage:
                    usage["prompt_tokens"] = getattr(response.usage, "prompt_tokens", None)
                    usage["completion_tokens"] = getattr(response.usage, "completion_tokens", None)
                    usage["total_tokens"] = getattr(response.usage, "total_tokens", None)
                print("OpenAI answer generated successfully.")
            except openai.APIKeyMissingError:
                print("Error: OpenAI API Key is missing.")
                answer = "Error: OpenAI API Key is missing. Please set it."
            except openai.AuthenticationError:
                print("Error: OpenAI authentication failed. Check API key.")
                answer = "Error: Invalid OpenAI API Key."
            except openai.RateLimitError:
                 print("Error: OpenAI rate limit exceeded.")
                 answer = "Error: OpenAI rate limit exceeded. Please try again later."
            except Exception as e:
                print(f"Error during OpenAI inference with model {llm_model}: {e}")
                answer = f"Error generating answer with OpenAI: {e}"
        else:
            print("Error: OpenAI client not available. Cannot use OpenAI models.")
            answer = "Error: OpenAI client not initialized. Check API key."

    # After Ollama inference, approximate usage if not set
    if llm_model.startswith("ollama:"):
        # Approximate prompt and completion token counts by word count
        prompt_tokens = len(prompt.split())
        completion_tokens = len(answer.split())
        usage["prompt_tokens"] = prompt_tokens
        usage["completion_tokens"] = completion_tokens
        usage["total_tokens"] = prompt_tokens + completion_tokens

    return answer, context_text, usage  # Return generated answer, context, and usage info
