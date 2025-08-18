# main.py

from graph import graph
from langchain_core.messages import BaseMessage

THREAD_ID = "user-session-1"

def _short(text: str, n: int = 220) -> str:
    if not text:
        return ""
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "…"

def _msg_role(msg) -> str:
    if not isinstance(msg, dict):
        return getattr(msg, "type", None) or "unknown"
    return msg.get("role") or "unknown"

def _msg_content(msg):
    if isinstance(msg, dict):
        return msg.get("content")
    return getattr(msg, "content", None)

def _msg_tool_calls(msg):
    if isinstance(msg, dict):
        ak = msg.get("additional_kwargs") or {}
        return msg.get("tool_calls") or ak.get("tool_calls")
    # LangChain AIMessage keeps tool calls on .tool_calls or .additional_kwargs["tool_calls"]
    tc = getattr(msg, "tool_calls", None)
    if tc:
        return tc
    ak = getattr(msg, "additional_kwargs", {}) or {}
    return ak.get("tool_calls")

def _describe_last_message(state):
    msgs = state.get("messages", [])
    if not msgs:
        return "(no messages)"
    last = msgs[-1]
    role = _msg_role(last)
    tool_calls = _msg_tool_calls(last)
    if tool_calls:
        items = []
        for tc in tool_calls:
            name = tc.get("name") or (tc.get("function") or {}).get("name") or "tool"
            args = tc.get("args") or (tc.get("function") or {}).get("arguments")
            items.append(f"{name}({args})")
        return f"{role} → tool_calls: " + ", ".join(items)
    return f"{role}: {_short(_msg_content(last))}"

def stream_with_debug(user_input: str):
    init_state = {"messages": [{"role": "user", "content": user_input}]}
    config = {"configurable": {"thread_id": THREAD_ID}}

    print("\n--- Graph execution trace ---")
    step = 0
    final_state = None
    for event in graph.stream(init_state, config):
        for node_name, state in event.items():
            step += 1
            print(f"[{step:02d}] node = {node_name:8s} | {_describe_last_message(state)}")
            final_state = state  # keep last seen state
    print("--- end trace ---\n")

    # Prefer the final streamed state if we have it; else get a fresh final result
    if final_state and final_state.get("messages"):
        last = final_state["messages"][-1]
    else:
        result = graph.invoke(init_state, config)
        last = result["messages"][-1]

    content = _msg_content(last)
    print("Assistant:", content or "")

def main():
    # ascii layout
    try:
        print("\nGraph layout:\n")
        print(graph.get_graph().draw_ascii())
        print()
    except Exception:
        pass

    print("Chatbot ready (type 'quit' to exit).")
    while True:
        prompt = input("You: ")
        if prompt.lower() in {"quit", "exit", "q"}:
            break
        stream_with_debug(prompt)

if __name__ == "__main__":
    main()
