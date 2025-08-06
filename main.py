# main.py

from graph import graph

# 1️⃣ Choose a thread ID for this conversation session.
THREAD_ID = "user-session-1"

def stream_graph_updates(user_input: str):
    # Prepare initial state and configuration
    init_state = {"messages": [{"role": "user", "content": user_input}]}
    config = {"configurable": {"thread_id": THREAD_ID}}

    # Stream responses from the graph
    for event in graph.stream(init_state, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def main():
    print("Chatbot is ready!")
    print("(Type quit/exit/q to stop.)\n")
    while True:
        prompt = input("You: ")
        if prompt.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        stream_graph_updates(prompt)

if __name__ == "__main__":
    main()
