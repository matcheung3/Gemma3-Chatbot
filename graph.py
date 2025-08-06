# graph.py

from typing import Annotated, TypedDict

# LangGraph core
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# Ollama LLM
from langchain_ollama import ChatOllama

# 1️⃣ Define conversation state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2️⃣ Initialize the graph builder
graph_builder = StateGraph(State)

# 3️⃣ Load the local Ollama model
llm = ChatOllama(
    model="gemma3:4b-it-qat",
    temperature=0.7,
)

# 4️⃣ Define the chatbot node
def chatbot(state: State) -> dict:
    # Send the accumulated messages to the model and get a response
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 5️⃣ Assemble the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")

# 6️⃣ Compile with in-memory checkpointing
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)