# etl/amazon_scrapper.py
# Web scraper for Amazon Australia (amazon.com.au).
# Uses undetected_chromedriver to bypass bot detection, and BeautifulSoup
# to parse product listings and reviews from search result pages.
# Outputs structured data: [product_id, title, rating, total_reviews, price, reviews]

import csv
import time
import re
import os
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class AmazonAUScraper:
    """Scraper for Amazon Australia (amazon.com.au) product listings and reviews."""

    BASE_URL = "https://www.amazon.com.au"

    def __init__(self,output_dir="data"):
        # Directory where CSV output and debug files are saved
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _create_driver(self):
        """Create and return a configured undetected Chrome driver.
        Uses anti-detection options to mimic a real browser session."""
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")  # Hide automation flags
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-dev-shm-usage")   # Prevent shared memory issues in containers
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")    # Realistic viewport size
        options.add_argument("--lang=en-AU")               # Match Amazon AU locale
        # Realistic User-Agent string to avoid bot detection
        options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36")
        # Launch Chrome via subprocess with a specific version to match chromedriver
        driver = uc.Chrome(options=options, use_subprocess=True, version_main=148)
        return driver


    def _dismiss_popups(self, driver):
        """Dismiss common Amazon popups (cookie consent, location banners, etc.).
        Tries multiple CSS selectors — silently skips if none are found."""
        popup_selectors = [
            "input#sp-cc-accept",                        # Cookie consent button
            "span.a-button-text[data-action-type='DISMISS']",  # Dismissable banners
            "button[data-action-type='DISMISS']",
            "#nav-main .nav-a[data-nav-role='close']",   # Navigation close button
        ]
        for sel in popup_selectors:
            try:
                el = driver.find_element(By.CSS_SELECTOR, sel)
                el.click()
                time.sleep(0.5)
            except Exception:
                continue  # Popup not present — that's fine

    def get_top_reviews(self, product_url, count=2):
        """Fetch the top customer reviews from an Amazon AU product page.

        Opens the product URL in a new Chrome instance, scrolls to load
        reviews, and extracts review text using multiple CSS selectors
        (Amazon frequently changes their class names).

        Args:
            product_url: Full URL to the Amazon product page.
            count: Number of reviews to retrieve.

        Returns:
            A string of reviews separated by ' || ', or 'No reviews found'.
        """
        driver = self._create_driver()

        # Validate URL before attempting to navigate
        if not product_url.startswith("http"):
            driver.quit()
            return "No reviews found"

        try:
            driver.get(product_url)
            time.sleep(4)
            self._dismiss_popups(driver)

            # Scroll down multiple times to trigger lazy-loading of review sections
            for _ in range(5):
                ActionChains(driver).send_keys(Keys.END).perform()
                time.sleep(1.5)

            # Parse the fully-loaded page source with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, "html.parser")

            seen = set()     # Track seen reviews to avoid duplicates
            reviews = []

            # --- Primary approach: extract review body text directly ---
            # Try multiple selectors (Amazon changes these frequently)
            review_bodies = soup.select("span[data-hook='review-body']")
            if not review_bodies:
                review_bodies = soup.select("div.reviewText")
            if not review_bodies:
                review_bodies = soup.select("div.a-expander-content.reviewText")
            if not review_bodies:
                review_bodies = soup.select("span.review-text-content")

            for body in review_bodies:
                text = body.get_text(separator=" ", strip=True)
                # Filter out very short snippets (likely not real reviews)
                if text and len(text) > 20 and text not in seen:
                    reviews.append(text)
                    seen.add(text)
                if len(reviews) >= count:
                    break

            # --- Fallback: try parent review containers ---
            if not reviews:
                review_blocks = soup.select("div[data-hook='review']")
                if not review_blocks:
                    review_blocks = soup.select("div[id^='customer_review-']")
                if not review_blocks:
                    review_blocks = soup.select("div.review")

                for block in review_blocks:
                    text = block.get_text(separator=" ", strip=True)
                    if text and len(text) > 20 and text not in seen:
                        reviews.append(text)
                        seen.add(text)
                    if len(reviews) >= count:
                        break

            # --- Last resort: try the #customerReviews section ---
            if not reviews:
                cr_section = soup.select_one("#customerReviews, #reviewsMedley")
                if cr_section:
                    for p in cr_section.select("span[data-hook='review-body'], div.reviewText"):
                        text = p.get_text(separator=" ", strip=True)
                        if text and len(text) > 20 and text not in seen:
                            reviews.append(text)
                            seen.add(text)
                        if len(reviews) >= count:
                            break

        except Exception as e:
            print(f"[DEBUG] Error fetching reviews: {e}")
            reviews = []

        driver.quit()
        # Join multiple reviews with ' || ' separator
        return " || ".join(reviews) if reviews else "No reviews found"

    def scrape_amazon_products(self, query, max_products=1, review_count=2):
        """Scrape Amazon AU search results for a given query.

        Opens the search results page, extracts product details (title, price,
        rating, review count, ASIN), then visits each product page to fetch reviews.

        Args:
            query: Search term for products (e.g. "iPhone 15").
            max_products: Maximum number of products to scrape from results.
            review_count: Number of reviews to fetch per product.

        Returns:
            List of lists: [product_id, title, rating, total_reviews, price, top_reviews]
        """
        driver = self._create_driver()

        # Build the Amazon AU search URL (spaces → '+')
        search_url = f"{self.BASE_URL}/s?k={query.replace(' ', '+')}"
        print(f"[DEBUG] Navigating to: {search_url}")
        
        # Navigate to the search results page
        try:
            driver.get(search_url)
            time.sleep(5)       # Wait for page to fully load
            self._dismiss_popups(driver)
            time.sleep(2)
        except Exception as e:
            print(f"[DEBUG] Error loading search page: {e}")
            try:
                driver.quit()
            except Exception:
                pass
            return []

        products = []

        # Grab the page source — may fail if Amazon closed the browser (anti-bot)
        try:
            page_source = driver.page_source
        except Exception as e:
            print(f"[DEBUG] Browser window was closed by Amazon anti-bot detection: {e}")
            print("[DEBUG] Tip: Amazon may be blocking automated access. Try again in a few minutes.")
            try:
                driver.quit()
            except Exception:
                pass
            return []

        # Find product items on the page using Amazon's data-asin attribute
        try:
            # Primary selector: search result items with ASIN
            items = driver.find_elements(By.CSS_SELECTOR, "div[data-asin][data-component-type='s-search-result']")
            if not items:
                # Fallback selector
                items = driver.find_elements(By.CSS_SELECTOR, "div.s-result-item[data-asin]")
            # Filter out items with empty data-asin (ad spacers, sponsored slots)
            items = [item for item in items if item.get_attribute("data-asin")]
        except Exception as e:
            print(f"[DEBUG] Browser window closed while finding products: {e}")
            try:
                driver.quit()
            except Exception:
                pass
            return []

        print(f"[DEBUG] Found {len(items)} product items on page")

        # If no items found, save the page HTML for debugging
        if not items:
            debug_path = os.path.join(self.output_dir, "debug_amazon_page.html")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(page_source)
            print(f"[DEBUG] No items found. Page source saved to {debug_path}")

        # Process only up to max_products items
        items = items[:max_products]
        for item in items:
            try:
                # Extract the Amazon Standard Identification Number (ASIN)
                asin = item.get_attribute("data-asin")
                product_id = asin if asin else "N/A"

                # --- Extract product title ---
                # Try multiple CSS selectors (Amazon changes class names frequently)
                title = None
                for sel in [
                    "h2 span",
                    "h2 a span",
                    "h2 span.a-text-normal",
                    "span.a-size-medium.a-color-base.a-text-normal",
                    "span.a-size-base-plus.a-color-base.a-text-normal",
                ]:
                    try:
                        title = item.find_element(By.CSS_SELECTOR, sel).text.strip()
                        if title:
                            break
                    except Exception:
                        continue

                # --- Extract price ---
                price = None
                for sel in [
                    "span.a-price span.a-offscreen",   # Hidden price text (most reliable)
                    "span.a-price-whole",                # Visible price digits
                    "span.a-color-base span.a-offscreen",
                ]:
                    try:
                        price = item.find_element(By.CSS_SELECTOR, sel).text.strip()
                        if price:
                            break
                    except Exception:
                        continue

                # --- Extract star rating ---
                rating = None
                for sel in [
                    "span.a-icon-alt",                   # "4.5 out of 5 stars" alt text
                    "i.a-icon-star-small span.a-icon-alt",
                ]:
                    try:
                        rating_text = item.find_element(By.CSS_SELECTOR, sel).text.strip()
                        if rating_text:
                            # Extract the numeric rating (e.g. "4.5" from "4.5 out of 5 stars")
                            match = re.search(r"[\d.]+", rating_text)
                            rating = match.group(0) if match else rating_text
                            break
                    except Exception:
                        continue

                # --- Extract total review count ---
                total_reviews = "N/A"
                for sel in [
                    "span.a-size-base.s-underline-text",
                    "a[href*='customerReviews'] span",
                    "span.a-size-base",
                ]:
                    try:
                        el = item.find_element(By.CSS_SELECTOR, sel)
                        text = el.text.strip()
                        if text and re.search(r"[\d,]+", text):
                            # Extract just the number (e.g. "1,234" from "1,234 ratings")
                            match = re.search(r"[\d,]+", text)
                            total_reviews = match.group(0) if match else "N/A"
                            break
                    except Exception:
                        continue

                # --- Build product link from ASIN (most reliable method) ---
                # Direct /dp/ links constructed from ASIN are more reliable than
                # extracting href attributes (which may be redirect/tracking URLs)
                product_link = f"{self.BASE_URL}/dp/{asin}" if asin else None

                if not product_link:
                    # Fallback: try finding any link with /dp/ in the href
                    try:
                        link_el = item.find_element(By.CSS_SELECTOR, "a[href*='/dp/']")
                        href = link_el.get_attribute("href")
                        if href:
                            product_link = href if href.startswith("http") else self.BASE_URL + href
                    except Exception:
                        product_link = "N/A"

                # Skip items without a title (likely ads or layout elements)
                if not title:
                    print(f"[DEBUG] Skipping item ASIN={asin} - no title found.")
                    continue

                print(f"[DEBUG] Found product: {title[:50]}... | Price: {price} | Rating: {rating}")

            except Exception as e:
                print(f"Error occurred while processing item: {e}")
                continue

            # Navigate to the product page and fetch customer reviews
            top_reviews = "No reviews found"
            if product_link and product_link != "N/A" and "amazon.com.au" in product_link:
                top_reviews = self.get_top_reviews(product_link, count=review_count)

            # Append the complete product record
            products.append([product_id, title, rating, total_reviews, price, top_reviews])

        driver.quit()
        return products

    def save_to_csv(self, data,filename="amazon_product_review.csv"):
        """Save the scraped product data to a CSV file with standard column headers."""
        # Resolve the output file path
        if os.path.isabs(filename):
            path = filename
        elif os.path.dirname(filename):
            # Filename includes a subdirectory (e.g. 'data/product_reviews.csv')
            path = filename
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            # Plain filename — save to the output directory
            path = os.path.join(self.output_dir, filename)

        # Write CSV with header row
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["product_id", "product_title", "rating", "total_reviews", "price", "top_reviews"])
            writer.writerows(data)
        print(f"[DEBUG] Data saved to {path}")