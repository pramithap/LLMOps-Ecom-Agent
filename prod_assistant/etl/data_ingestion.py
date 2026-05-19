import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from prod_assistant.utils.modal_loader import ModelLoader
from prod_assistant.utils.config_loader import load_config
from prod_assistant.logger import GLOBAL_LOGGER as log

class DataIngestion:

    def __init__(self):
        """
        Initializes the DataIngestion class by loading environment variables and configuration settings."""

        print("Initialize the ingestion pipeline")
        self.model_loader = ModelLoader()  # Ensure models and API keys are loaded at startup
        self._load_env_variables()  # Load .env variables (API keys, etc.)
        self.csv_path = self._get_csv_path()  # Resolve path to product_reviews.csv
        self.product_data = self._load_csv()  # Load the scraped CSV data into a DataFrame
        self.config = load_config()  # Load YAML config for any additional settings
        log.info("DataIngestion initialized", config_keys=list(self.config.keys()))

    def _load_env_variables(self):
        load_dotenv()  # Load environment variables from .env file at project root

        #These  env vars are required for connecting to AstraDB
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]

        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables:{missing_vars}")
        
        # Store connection credentials for later use in store_in_vector_db()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    def _get_csv_path(self):
        """Locate the product_reviews.csv file in the data/ directory.
        Uses CWD-relative path (expects to be run from the project root)."""
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir, 'data', 'product_reviews.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        return csv_path


    def _load_csv(self):
        """Load the CSV into a pandas DataFrame and validate required columns."""
        df = pd.read_csv(self.csv_path)

        #Ensure the CSV has all teh columns we expect before proceeding
        expected_columns = {'product_id', 'product_title', 'rating', 'total_reviews', 'price', 'top_reviews'}
        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {expected_columns}")

        return df

    @staticmethod
    def _sanitize_value(value):
        """Replace NaN/None with empty string for JSON compatibility.
        AstraDB metadata values must be valid JSON (no NaN/None)."""
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return value

    def transform_data(self):
        """Transform each row of the product DataFrame into a LangChain Document.
        Document structure:
        - page_content: the product's review text (what gets embedded for search)
        - metadata: product_id, title, rating, total_reviews, price
        (stored alongside the vector for filtering/display)
        """
        product_list = []

        for _, row in self.product_data.iterrows():
            product_entry = {
                    "product_id": self._sanitize_value(row["product_id"]),
                    "product_title": self._sanitize_value(row["product_title"]),
                    "rating": self._sanitize_value(row["rating"]),
                    "total_reviews": self._sanitize_value(row["total_reviews"]),
                    "price": self._sanitize_value(row["price"]),
                    "top_reviews": self._sanitize_value(row["top_reviews"])
                }
            product_list.append(product_entry)

        # Convert dicts into LangChain Document objects
        documents = []
        for entry in product_list:
            # Metadata = structured fields for filtering and display in responses
            metadata = {
                    "product_id": entry["product_id"],
                    "product_title": entry["product_title"],
                    "rating": entry["rating"],
                    "total_reviews": entry["total_reviews"],
                    "price": entry["price"]
            }
            # page_content = the text that will be embedded and searched against
            # Reviews are the most semantically rich content for similarity search
            page_content = entry["top_reviews"] if entry["top_reviews"] else "No reviews available"
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)

        print(f"Transformed {len(documents)} documents.")
        return documents

    def store_in_vector_db(self, documents: List[Document]):
        """Store the transformed documents into AstraDB vector store.
        
        Creates embeddings using the configured model and inserts docs
        along with their metadata. Returns the vector store instance
        and the list of inserted document IDs.
        """
        # Read collection name from config (e.g. "ecommercedata")
        collection_name = self.config["astra_db"]["collection_name"]

        # Connect to AstraDB with the embedding model
        vstore = AstraDBVectorStore(
            embedding=self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
        )

        # Insert all documents (embeddings are computed automatically)
        inserted_ids = vstore.add_documents(documents)
        print(f"Successfully inserted {len(inserted_ids)} documents into AstraDB.")
        return vstore, inserted_ids



    def run_pipeline(self):
        """Run the full ingestion pipeline: CSV → Documents → AstraDB.
        Also performs a quick sanity-check search to verify the data was stored."""
        
        # Step 1: Transform CSV data into LangChain Documents
        documents = self.transform_data()

        # Step 2: Store documents in AstraDB (creates embeddings + inserts)
        vstore, inserted_ids = self.store_in_vector_db(documents)

        # Step 3: Quick verification search to confirm data is queryable
        query = "Can you tell me the low budget iphone?"
        results = vstore.similarity_search(query)

        print(f"\nSample search results for query: '{query}'")
        for res in results:
            print(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")



#-- Run the full pipeline when executed directly --#
if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run_pipeline()