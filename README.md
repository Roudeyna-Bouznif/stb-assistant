# STB Assistant — AI banking FAQ & KYC assistant (prototype)

RAG-based chatbot prototype built with FastAPI, ChromaDB and Gemini.
It answers banking FAQ and KYC procedure questions from a knowledge base,
cites its sources, and falls back to a safe message when it has no reliable answer.

All documents in `data/` are fictional. No real bank data is used.

## Architecture
- `data/`: knowledge base (Markdown files)
- `app/rag.py`: chunking, embeddings, vector search (ChromaDB)
- `app/main.py`: FastAPI API with a `/chat` route
- `web/`: chat widget (HTML/CSS/JS), coming next

## Setup
1. `python -m venv venv`, then activate it (`venv\Scripts\activate` on Windows)
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Gemini API key
4. Index the documents: `python -m app.rag`
5. Run the API: `uvicorn app.main:app --reload`
6. Test it at http://127.0.0.1:8000/docs