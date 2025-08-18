# graph.py
from typing import Annotated, TypedDict
import json
import re

# LangGraph core
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# LLM (Ollama — tools-capable Gemma3)
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

# Tools
from rag_tool import rag_tool                 # @tool("rag_search")
from pdf_image_tool import vision_pdf_search  # @tool("vision_pdf_search")


# ---------- State ----------
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# ---------- LLMs ----------
llm = ChatOllama(
    model="PetrosStav/gemma3-tools:4b",
    temperature=0.7,
)
llm_with_tools = llm.bind_tools([rag_tool, vision_pdf_search])

# ---------- Policy (nudge tool usage) ----------
POLICY = (
    "You are a helpful assistant with two tools:\n"
    "1) rag_search — use for questions that likely depend on local docs (txt/md/pdf text).\n"
    "2) vision_pdf_search — use when the content appears to be scanned or image-heavy "
    "(figures/tables) or when text retrieval returns nothing.\n"
    "If unsure or not 100% certain, call a tool first, then answer grounded in the returned CONTEXT."
)

def policy_node(state: State) -> dict:
    return {"messages": [{"role": "system", "content": POLICY}]}


# ---------- Helpers ----------
# Match anything inside ```tool_code ... ```
_TOOL_FENCE = re.compile(r"```tool_code\s*(.+?)\s*```", re.DOTALL)

def _extract_tool_invocation_from_fence(text: str):
    """
    Supports:
      1) JSON:  ```tool_code {"name":"rag_search","parameters":{"query":"..."} } ```
      2) Code:  ```tool_code rag_search(query="...") ```
                ```tool_code rag_search("...") ```
                ```tool_code print(rag_search.search_docs(query="...")) ```
                ```tool_code vision_pdf_search(query="...") ```
    Returns {"name": <tool_name>, "args": {"query": "..."} } or None.
    """
    if not text or not isinstance(text, str):
        return None
    m = _TOOL_FENCE.search(text)
    if not m:
        return None
    body = m.group(1).strip()

    # Try JSON first
    try:
        data = json.loads(body)
        name = data.get("name")
        params = data.get("parameters", {}) or {}
        if name:
            return {"name": name, "args": params}
    except Exception:
        pass

    # Parse code-like patterns
    patterns = [
        (r'(?:^|[\s;])rag_search(?:\.\w+)?\s*\(\s*(?:query\s*=\s*)?[\'"]([^\'"]+)[\'"]\s*\)', "rag_search"),
        (r'(?:^|[\s;])vision_pdf_search(?:\.\w+)?\s*\(\s*(?:query\s*=\s*)?[\'"]([^\'"]+)[\'"]\s*\)', "vision_pdf_search"),
    ]
    for pat, tool_name in patterns:
        mc = re.search(pat, body)
        if mc:
            return {"name": tool_name, "args": {"query": mc.group(1)}}
    return None

def _has_tool_fence(text: str) -> bool:
    # Detects a fenced block even if it’s malformed/empty
    return bool(_TOOL_FENCE.search(text or ""))

_TOOL_REGISTRY = {
    "rag_search": rag_tool,
    "vision_pdf_search": vision_pdf_search,
}

def _last_user_text(msgs: list) -> str:
    for m in reversed(msgs or []):
        role = getattr(m, "type", None) or getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        if role in ("human", "user"):
            return content or ""
    return ""

def _vision_if_empty(query: str, rag_out: str) -> str:
    # If text RAG found nothing, try vision on PDF images (and don't crash if vision fails)
    if "No relevant context found" in (rag_out or ""):
        try:
            vis_out = vision_pdf_search.invoke({"query": query})
        except Exception as e:
            return f"CONTEXT from rag_search (vision failed: {e}):\n{rag_out}"
        return f"CONTEXT from vision_pdf_search:\n{vis_out}"
    return f"CONTEXT from rag_search:\n{rag_out}"

def _run_tool_and_respond(state_msgs: list, tool_name: str, args: dict) -> dict:
    """Execute the tool locally and return a final LLM response using CONTEXT as a system message."""
    if tool_name == "rag_search":
        out = rag_tool.invoke(args if isinstance(args, dict) else {"query": str(args)})
        ctx = _vision_if_empty(_last_user_text(state_msgs), str(out))
    elif tool_name == "vision_pdf_search":
        out = vision_pdf_search.invoke(args if isinstance(args, dict) else {"query": str(args)})
        ctx = f"CONTEXT from vision_pdf_search:\n{out}"
    else:
        return {"messages": [AIMessage(content=f"(Unknown tool '{tool_name}')", role="assistant")]}
    final = llm.invoke(state_msgs + [{"role": "system", "content": ctx}])
    return {"messages": [final]}


# ---------- Chatbot (handles tool_calls, fenced code/JSON, fence-only fallback, and blank fallback) ----------
def chatbot(state: State) -> dict:
    resp = llm_with_tools.invoke(state["messages"])

    if isinstance(resp, AIMessage) and getattr(resp, "content", None):
        # A) Structured tool_calls (OpenAI-style)
        tool_calls = getattr(resp, "tool_calls", None) or getattr(resp, "additional_kwargs", {}).get("tool_calls")
        if tool_calls:
            call = tool_calls[0]
            name = call.get("name") or (call.get("function") or {}).get("name")
            args = call.get("args") or (call.get("function") or {}).get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"query": args}
            return _run_tool_and_respond(state["messages"], name, args)

        # B) Fenced tool_code
        tc = _extract_tool_invocation_from_fence(resp.content)
        if tc:
            return _run_tool_and_respond(state["messages"], tc["name"], tc["args"])

        # NEW: fence present but no valid call → soft fallback (RAG then maybe vision)
        if _has_tool_fence(resp.content):
            user_q = _last_user_text(state.get("messages", []))
            if user_q:
                rag_out = rag_tool.invoke({"query": user_q})
                ctx = _vision_if_empty(user_q, str(rag_out))
                final = llm.invoke(state["messages"] + [{"role": "system", "content": ctx}])
                return {"messages": [final]}
            return {"messages": [resp]}

        # C) Plain text answer
        return {"messages": [resp]}

    # D) Blank or non-AIMessage → soft fallback: RAG then maybe vision
    user_q = _last_user_text(state.get("messages", []))
    if user_q:
        rag_out = rag_tool.invoke({"query": user_q})
        ctx = _vision_if_empty(user_q, str(rag_out))
        final = llm.invoke(state["messages"] + [{"role": "system", "content": ctx}])
        return {"messages": [final]}

    return {"messages": [AIMessage(content="I couldn't generate a response. Try rephrasing your question.")]}  # type: ignore[arg-type]


# ---------- Tool node (kept so native structured tool_calls still route) ----------
tool_node = ToolNode(tools=[rag_tool, vision_pdf_search])

# ---------- Wiring ----------
graph_builder.add_node("policy", policy_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("policy")
graph_builder.add_edge("policy", "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)   # native structured calls → ToolNode
graph_builder.add_edge("tools", "chatbot")                        # return to LLM after tools

# ---------- Compile ----------
graph = graph_builder.compile(checkpointer=InMemorySaver())
