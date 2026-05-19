# Centralised prompt registry for the e-commerce product assistant.
# Each prompt is stored as a PromptTemplate with validation, versioning,
# and a description. The PROMPT_REGISTRY dict maps PromptType enum values
# to their corresponding PromptTemplate instances.


from enum import Enum
from typing import Dict
import string

# Enum for identifying prompt types across the codebase
# Add new members here when creating new bot personalities or use-cases
class PromptType(str, Enum):
    PRODUCT_BOT = "product_bot"
    # REVIEW_BOT = "review_bot"
    # COMPARISON_BOT = "comparison_bot"


class PromptTemplate:
    """Wrapper around a prompt string with placeholder validation and metadata."""

    def __init__(self, template: str, description: str, version: str = "v1"):
        self.template = template.strip() # Remove leading/trailing whitespace
        self.description = description  # Human-readable purpose of this prompt
        self.version = version          # Track prompt iterations

    def format(self, **kwargs) -> str:
        """Format the prompt with provided keyword arguments after validating placeholders."""
        # Check all {placeholder} fields in the template provided
        missing = [
            f for f in self.required_placeholders() if f not in kwargs
        ]
        if missing:
            raise ValueError(f"Missing placeholders for prompt formatting: {missing}")
        return self.template.format(**kwargs)

    def required_placeholders(self):
        """Extract all {field_name} placeholders from the template striong.
        uses Python' sstring.formatter to parse the template and find all unique field names."""
        return [field_name for _, field_name, _, _ in string.Formatter(). parse(self.template) if field_name]
    

# --------Central Prompt Registry -------- #
# Maps each PromptType to its corresponding PromptTemplate instance.

PROMPT_REGISTRY: Dict[PromptType, PromptTemplate] = {
    PromptType.PRODUCT_BOT: PromptTemplate(
        """
        You are an expert EcommerceBot specialized in product recommendations and handling customer queries.
        Analyze the provided product titles, ratings, prices, and reviews to provide accurate, helpful responses.
        Stay relevant to the context, and keep your answers concise and informative.
        
        IMPORTANT: Always include the product price in your response when available in the context.
        If the user asks about price, make sure to extract and display the exact price from the context.

        CONTEXT:
        {context}

        QUESTION: {question}

        YOUR ANSWER:
        """,
        description="Handles ecommerce QnA & product recommendation flows"
    )
}