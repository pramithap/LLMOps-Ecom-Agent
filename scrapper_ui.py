# =============================================================================
# Streamlit UI for the Product Review Scraper
# Provides a web interface to scrape product listings from Flipkart or Amazon AU,
# save results to CSV, and optionally ingest into AstraDB for RAG retrieval.
# Run with: streamlit run scrapper_ui.py
# =============================================================================

import streamlit as st
# from prod_assistant.etl.data_scrapper import FlipkartScraper    # Flipkart scraper
from prod_assistant.etl.amazon_scraper import AmazonAUScraper  # Amazon AU scraper
import os

# Initialize scraper instances and set the default CSV output path
# flipkart_scraper = FlipkartScraper()
amazon_scraper = AmazonAUScraper()
output_path = "data/product_reviews.csv"

# Page title
st.title("📦 Product Review Scraper")

# Initialize the dynamic product input list in session state
# (Streamlit reruns the script on every interaction, so we persist state)
if "product_inputs" not in st.session_state:
    st.session_state.product_inputs = [""]


def add_product_input():
    """Callback: appends an empty string to the input list, creating a new text field."""
    st.session_state.product_inputs.append("")


# --- Platform selection dropdown ---
st.subheader("🌐 Select Platform")
platform = st.selectbox("Choose the e-commerce platform to scrape:", ["Flipkart", "Amazon AU"], index=0)

# --- Optional free-text description used as an additional search keyword ---
st.subheader("📝 Optional Product Description")
product_description = st.text_area("Enter product description (used as an extra search keyword):")

# --- Dynamic product name inputs (user can add more with the button) ---
st.subheader("🛒 Product Names")
updated_inputs = []
for i, val in enumerate(st.session_state.product_inputs):
    input_val = st.text_input(f"Product {i+1}", value=val, key=f"product_{i}")
    updated_inputs.append(input_val)
st.session_state.product_inputs = updated_inputs

# Button to add another product input field
st.button("➕ Add Another Product", on_click=add_product_input)

# --- Scraping parameters ---
max_products = st.number_input("How many products per search?", min_value=1, max_value=10, value=1)
review_count = st.number_input("How many reviews per product?", min_value=1, max_value=10, value=2)

# --- Main scraping action ---
if st.button("🚀 Start Scraping"):
    # Collect non-empty product inputs
    product_inputs = [p.strip() for p in st.session_state.product_inputs if p.strip()]
    # Append the optional description as an additional search query
    if product_description.strip():
        product_inputs.append(product_description.strip())

    if not product_inputs:
        st.warning("⚠️ Please enter at least one product name or a product description.")
    else:
        final_data = []
        # Pick the correct scraping function based on selected platform
        scrape_fn = amazon_scraper.scrape_amazon_products

        # Scrape each query term and accumulate results
        for query in product_inputs:
            st.write(f"🔍 Searching **{platform}** for: {query}")
            results = scrape_fn(query, max_products=max_products, review_count=review_count)
            final_data.extend(results)

        # Deduplicate by product title (keep first occurrence)
        unique_products = {}
        for row in final_data:
            if row[1] not in unique_products:
                unique_products[row[1]] = row

        final_data = list(unique_products.values())
        st.session_state["scraped_data"] = final_data  # Persist for the AstraDB button below

        # Save to CSV using the appropriate scraper's save method
        saver = amazon_scraper
        saver.save_to_csv(final_data, output_path)
        st.success(f"✅ Data saved to `{output_path}`")

        # Offer a download button for the generated CSV
        st.download_button("📥 Download CSV", data=open(output_path, "rb"), file_name="product_reviews.csv")

# --- Vector DB ingestion button (only shown after scraping completes) ---
# This button is outside the scraping block so it persists across reruns
if "scraped_data" in st.session_state and st.button("🧠 Store in Vector DB (AstraDB)"):
    with st.spinner("📡 Initializing ingestion pipeline..."):
        try:
            from prod_assistant.etl.data_ingestion import DataIngestion  # lazy import — avoids pydantic plugin scan at startup
            ingestion = DataIngestion()
            st.info("🚀 Running ingestion pipeline...")
            ingestion.run_pipeline()
            st.success("✅ Data successfully ingested to AstraDB!")
        except Exception as e:
            st.error("❌ Ingestion failed!")
            st.exception(e)