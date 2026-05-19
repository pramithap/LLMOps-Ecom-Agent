# evaluation/ragas_eval.py
# RAGAS (Retrieval-Augmented Generation Assessment) evaluation utilities.
# Provides two metrics:
#   1. Context Precision: How relevant are the retrieved docs to the question?
#   2. Response Relevancy: How relevant is the LLM's response to the question?
# Both use the same LLM (Gemini) to evaluate — runs async internally.

import asyncio
from prod_assistant.utils.model_loader import ModelLoader
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy

# Module-level model loader (shared across evaluation calls)
model_loader = ModelLoader()


def evaluate_context_precision(query, response, retrieved_context):
    """Evaluate how relevant the retrieved context is to the query.
    Uses LLMContextPrecisionWithoutReference — no ground truth needed.
    
    Args:
        query: The user's original question
        response: The LLM's generated response
        retrieved_context: List of context strings from the retriever
        
    Returns:
        Float score (0-1) or the exception if evaluation fails.
    """
    try:
        # Package the query, response, and contexts into a RAGAS sample
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context,
        )

        async def main():
            # Wrap our LLM in RAGAS's LangChain adapter
            llm = model_loader.load_llm()
            evaluator_llm = LangchainLLMWrapper(llm)
            # Create the context precision scorer and compute the score
            context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
            result = await context_precision.single_turn_ascore(sample)
            return result

        # Run the async evaluation in a new event loop
        return asyncio.run(main())
    except Exception as e:
        # Return the exception instead of raising — caller can check type
        return e

def evaluate_response_relevancy(query, response, retrieved_context):
    """Evaluate how relevant the LLM's response is to the original query.
    Uses both LLM and embeddings to compute a semantic relevancy score.
    
    Args:
        query: The user's original question
        response: The LLM's generated response
        retrieved_context: List of context strings from the retriever
        
    Returns:
        Float score (0-1) or the exception if evaluation fails.
    """
    try:
        # Package inputs into a RAGAS sample
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context,
        )

        async def main():
            # Wrap both LLM and embedding model in RAGAS adapters
            llm = model_loader.load_llm()
            evaluator_llm = LangchainLLMWrapper(llm)
            embedding_model = model_loader.load_embeddings()
            evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)
            # Create the response relevancy scorer (needs both LLM + embeddings)
            scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
            result = await scorer.single_turn_ascore(sample)
            return result

        # Run the async evaluation in a new event loop
        return asyncio.run(main())
    except Exception as e:
        return e