
import os
import sys
from pathlib import Path

# Add the prod_assistant directory to the Python path so this file can be
# run directly as a script (not just as part of the package)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_astradb import AstraDBVectorStore
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.config_loader import load_config
from dotenv import load_dotenv


class Retriever:
    """Manages the AstraDB vector Store connection and exposes a langchain retriever"""

    def __init__(self):
        # Load model factory and YAML config
        self.model_loader=ModelLoader()
        self.config=load_config()

        # Validate and store required environment variables
        self._load_env_variables()

        # These are lazily initialised on first call to load_retriever()
        self.vstore = None              # AstraDBVectorStore instance
        self.retriever_instance = None  # LangChain VectorStoreRetriever

    def _load_env_variables(self):
        """Load environment variables from .env file and validate all required vars are present."""
        load_dotenv()
         
        # List of env vars that must be set for AstraDB + embedding to work
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        
        # Check which required vars are missing
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        # Store connection details for use when creating the vector store
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
    
    def load_retriever(self):
        """Create (or return cached) AstraDB vector store and LangChain retriever.

        - Uses `autodetect_collection=True` so AstraDB auto-detects collection settings
        (embedding dimension, similarity metric, hybrid search capability).
        - Returns a similarity-based retriever with top_k from config.
        """

        if not self.vstore:
            #read collection name from config (e.g. "Ecommercedata")
            collection_name = self.config["astra_db"]["collection_name"]

            #Connect to AstraDB with explicit embeddings + auto detection
            # autodetect_colleciton = true - Enables hybrid seaerch (vector + lexical) 
            # if supported by the collection, otherwise falls back to vector search)

            self.vstore =AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                api_endpoint=self.db_api_endpoint,
                collection_name=collection_name,
                token=self.db_application_token,
                namespace=self.db_keyspace,
                autodetect_collection=True
            )

        if not self.retriever_instance:
            #Read top_k from config
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3

            # Create a LangChain retriever from the vector store
            # search_type="similarity" uses cosine similarity ranking
            # With autodetect, hybrid search (find_and_rerank) activates automatically
            self.retriever_instance = self.vstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": top_k}
                )
        return self.retriever_instance
    
    def call_retriever (self, query: str):
        """Convenience method to call the retriever directly with a query string."""
        retriever = self.load_retriever()
        output=retriever.invoke(query)
        return output
    

# ----Self-test code to verify the retriever can be loaded and called successfully ----

if __name__ == "__main__":
    user_query = "Can you suggest good budget iPhone under 1,00,00 INR?"
    
    # Create retriever and fetch documents
    retriever_obj = Retriever()
    retrieved_docs = retriever_obj.call_retriever(user_query)
    
    def _format_docs(docs) -> str:
        """Format a list of Document objects into a human-readable string
        with title, price, rating, and review content."""
        if not docs:
            return "No relevant documents found."
        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Reviews:\n{d.page_content.strip()}"
            )
            formatted_chunks.append(formatted)
        return "\n\n---\n\n".join(formatted_chunks)
    
    # Format docs for RAGAS evaluation input
    retrieved_contexts = [_format_docs([doc]) for doc in retrieved_docs]
    
    # Dummy response for testing the evaluation pipeline
    # (in production, this would come from the LLM generator)
    response="iphone 16 plus, iphone 16, iphone 15 are best phones under 1,00,000 INR."