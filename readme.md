# HR Support Chatbot (RAG)

This project is a Retrieval-Augmented Generation based HR support chatbot. It answers employee questions using indexed HR policy documents and a Streamlit chat interface.

## Features

- Streamlit chat UI for HR policy questions
- FAISS vector search over HR documents
- OpenAI embeddings for retrieval
- OpenAI or Groq selectable chat model in the sidebar
- Source preview for retrieved policy chunks
- Example HR questions and session controls

## Setup

Create and activate a virtual environment:

```powershell
C:\Users\laksh\anaconda3\python.exe -m venv virtual
.\virtual\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file from `.env_example`:

```env
OPENAI_API_KEY=""
GROQ_API_KEY=""
OPENAI_CHAT_MODEL="gpt-4o-mini"
GROQ_CHAT_MODEL=""
```

`OPENAI_API_KEY` is required for retrieval because the FAISS index uses OpenAI embeddings. `GROQ_API_KEY` is optional and enables Groq as the answer model.

## Index Documents

Place HR PDF files in the `documents/` folder, then run:

```powershell
python ingest.py
```

This creates the local `hr_faiss_index/` folder.

## Run

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Notes

- Do not commit `.env`; it contains API keys.
- The FAISS index is generated locally and excluded from git.
- Groq can be used for answers, but the existing retrieval index still needs OpenAI embeddings unless you rebuild the ingestion flow with another embedding provider.
