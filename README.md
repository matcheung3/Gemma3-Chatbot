# LangGraph & Ollama Gemma3 QAT Chatbot

This repository demonstrates a stateful chatbot built using **LangGraph** and **Ollama**, running the quantized Gemma3 model locally. Conversations are persisted in memory to maintain context between messages.

## Features

- **Stateful conversation** using LangGraph’s `StateGraph` API
- **Local inference** via Ollama (`gemma3:4b-it-qat`)
- **Streaming responses** with real-time output
- **In-memory checkpointing** to preserve chat history within a session

## Prerequisites

- **Python 3.12** or newer
- **Ollama CLI** installed and available in your PATH  
  ```bash
  ollama serve
  ```
- **Gemma3 4B QAT** model pulled:
  ```bash
  ollama pull gemma3:4b-it-qat
  ```

## Installation

1. **Clone the repository**  


2. **Create and activate a virtual environment** (recommended via Conda)  
   ```bash
   conda create -n gemma-chat python=3.12
   conda activate gemma-chat
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```text
langgraph-ollama-chatbot/
├── graph.py          # Defines the LangGraph StateGraph and LLM node
├── main.py           # Interactive loop to send and receive chat messages
├── requirements.txt  # Python package dependencies
└── README.md         # This file
```

## Usage

Run the chatbot from your terminal:
```bash
python main.py
```

Sample interaction:
```
You: Hi
Assistant: Hello! How can I help you today?

You: What's your name?
Assistant: I'm an AI chatbot powered by Gemma3 QAT and LangGraph.

You: quit
Assistant: Goodbye!
```

## Configuration

- To change the model, update the `model` parameter in `graph.py`:
  ```python
  llm = ChatOllama(model="gemma3:4b-it-qat")
  ```

- To adjust checkpoint behavior or thread ID, modify `main.py`:
  ```python
  THREAD_ID = "user-session-1"
  ```
