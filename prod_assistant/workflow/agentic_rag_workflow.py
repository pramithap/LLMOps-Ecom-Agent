from typing import Any, Dict, List, Optional, Tuple, Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import asyncio


from prod_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from prod_assistant.retrievar.retrieval import Retriever
from prod_assistant.utils.modal_loader import ModelLoader
from prod_assistant.evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy


class AgenticRAG:
    """Agentic RAG pipeline using LangGraph."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        rewrite_count: int

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

#------Helpers--------#
    def _format_docs(self,docs) -> str:
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


#------Nodes------#
    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        messages = state["messages"]
        last_message = str(messages[-1].content)

        if any(word in last_message.lower() for word in ["price", "review", "product"]):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message})
            return {"messages": [HumanMessage(content=response)]}

    def _vector_retriever(self, state: AgentState):
        print("--- RETRIEVER ---")
        query = str(state["messages"][-2].content)
        retriever = self.retriever_obj.load_retriever()
        docs = retriever.invoke(query)
        if not docs:
            print("--- RETRIEVER: No documents found ---")
            return {"messages": [HumanMessage(content="No relevant product data found in the database.")]}
        print(f"--- RETRIEVER: Found {len(docs)} documents ---")
        for i, d in enumerate(docs):
            print(f"  Doc {i}: metadata={d.metadata}")
        context = self._format_docs(docs)
        print(f"--- FORMATTED CONTEXT ---\n{context[:500]}...")
        return {"messages": [HumanMessage(content=context)]}

    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        # If no documents were found or rewrite limit reached, go straight to generator
        if "No relevant product data found" in docs:
            return "generator"
        
        rewrite_count = state.get("rewrite_count", 0)
        if rewrite_count >= 2:
            print("--- GRADER: Max rewrites reached, proceeding to generator ---")
            return "generator"

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})
        return "generator" if "yes" in score.lower() else "rewriter"

    def _generate(self, state: AgentState):
        print("--- GENERATE ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})
        return {"messages": [HumanMessage(content=response)]}

    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        question = state["messages"][0].content
        rewrite_count = state.get("rewrite_count", 0)
        new_q = self.llm.invoke(
            [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        )
        return {"messages": [HumanMessage(content=new_q.content)], "rewrite_count": rewrite_count + 1}


#-------- Build Workflow --------#

    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END},
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")
        return workflow

    def run(self, query: str,thread_id: str = "default_thread") -> str:
        """Run the workflow for a given query and return the final answer."""
        result = self.app.invoke({"messages": [HumanMessage(content=query)], "rewrite_count": 0},
                                config={"configurable": {"thread_id": thread_id}}) #Thread_id is unique to a chat conversation
        return result["messages"][-1].content


if __name__ == "__main__":
    
    
    rag_agent = AgenticRAG()
    answer = rag_agent.run("What is the price of iPhone 17?")
    print("\nFinal Answer:\n", answer)