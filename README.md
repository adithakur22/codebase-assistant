# 🤖 Codebase Assistant

An AI-powered assistant that answers questions about any codebase using RAG (Retrieval Augmented Generation).

## Features
- Upload any codebase folder and index it automatically
- Semantic search using HuggingFace embeddings + ChromaDB
- Natural language Q&A powered by LLaMA 3.1 via Groq
- Clean chat interface built with Gradio
- Supports Python, Java, JavaScript, TypeScript, and more

## Tech Stack
- LangChain
- ChromaDB (vector database)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Groq (LLM inference)
- Gradio (UI)

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `GROQ_API_KEY` to `.env`
4. Index your codebase: `python src/indexer.py`
5. Run the app: `python app.py`
