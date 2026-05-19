# utils/model_loader.py
# Responsible for loading LLM and embedding models based on YAML config + env vars.
# Supports multiple providers (Google Gemini, Groq) and centralises API key management.

import os
import sys
import json
from dotenv import load_dotenv
from prod_assistant.utils.config_loader import load_config

# LangChain wrappers for different LLM/embedding providers
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from prod_assistant.logger import GLOBAL_LOGGER as log
from prod_assistant.exception.custom_exception import ProductAssistantException
import asyncio


load_dotenv()  # Load environment variables from .env file at project root

class ApiKeyManager:
    """Reads all required API keys from environment variables at startup
    and logs which ones are present/missing for quick diagnostics."""

    def __init__(self):
    # Collect all API keys from the current environment
        self.api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
            "ASTRA_DB_API_ENDPOINT": os.getenv("ASTRA_DB_API_ENDPOINT"),
            "ASTRA_DB_APPLICATION_TOKEN": os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            "ASTRA_DB_KEYSPACE": os.getenv("ASTRA_DB_KEYSPACE"),
        }

        # Log the status of each key (present vs missing) — never log actual values
        for key, val in self.api_keys.items():
            if val:
                log.info(f"{key} loaded from environment")
            else:
                log.warning(f"{key} is missing from environment")

    def get(self, key: str):
        """Retrieve a specific API key by name."""
        return self.api_keys.get(key)
    
class ModelLoader:
    """
    Central factory for creating embedding and LLM instances.
    Reads provider/model settings from config.yaml and uses ApiKeyManager for auth.
    """
    def __init__(self):
        # Initialise key manager (validates env vars) and load YAML config
        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def load_embeddings(self):
        """
        Instantiate and return the Google Generative AI embedding model
        specified in config.yaml under `embedding_model.model_name`.
        """
        try:
            # Read the embedding model name from config (e.g. "models/gemini-embedding-001")
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)

            # Patch: ensure an asyncio event loop exists for gRPC async internals
            # (needed when running outside of an async context like a plain script)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            # Create and return the LangChain embeddings wrapper
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY")  # type: ignore
            )
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise ProductAssistantException("Failed to load embedding model", sys)

    def load_llm(self):
        """
        Instantiate and return the chat LLM based on the provider selected
        via the LLM_PROVIDER env var (defaults to "google").
        Supported providers: google (Gemini), groq (DeepSeek/Llama via Groq).
        """
        # Read the full LLM config block from YAML
        llm_block = self.config["llm"]

        # Determine which provider to use (env var overrides, default "google")
        provider_key = os.getenv("LLM_PROVIDER", "google")

        # Validate that the chosen provider exists in config.yaml
        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        # Extract provider-specific settings
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")       # e.g. "google", "groq"
        model_name = llm_config.get("model_name")    # e.g. "gemini-3-flash-preview"
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        # --- Provider-specific instantiation ---
        if provider == "google":
            # Google Gemini via LangChain's ChatGoogleGenerativeAI wrapper
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens
            )

        # elif provider == "groq":
        #     # Groq-hosted models (e.g. DeepSeek, Llama) via LangChain's ChatGroq wrapper
        #     return ChatGroq(
        #         model=model_name,
        #         api_key=self.api_key_mgr.get("GROQ_API_KEY"), #type: ignore
        #         temperature=temperature,
        #     )

        # elif provider == "openai":
        #     return ChatOpenAI(
        #         model=model_name,
        #         api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
        #         temperature=temperature
        #     )

        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")

# --- Quick self-test when run directly ---
if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")