# Network Intelligence Chat

Conversational agent that searches your combined LinkedIn network and answers natural language questions about connections.

## Overview

Import LinkedIn connections from multiple team members, ask questions in natural language, and get relevant profiles with ownership attribution — who on your team knows each person.

**Example queries:**
- "Who in our network works in fintech in Berlin?"
- "Find connections with ML experience who recently changed jobs"
- "Who can introduce me to investors in B2B SaaS?"

## Features

- **Multi-member network**: Combines LinkedIn connections from entire team
- **Smart retrieval**: SQL for exact lookups, semantic search for fuzzy queries, hybrid for complex questions
- **Ownership tracking**: Every result shows which team member knows that person
- **Grounded responses**: Citations to actual profiles, no hallucinations

## How to run
   Open the app on the following [link](network-intelligence-chat-production.up.railway.app). 
   (might be offline before presentation)


## How to run locally

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Set up environment**
```bash
cp .env.example .env
# Add your DATABASE_URL, MISTRAL_API_KEY, HF_TOKEN
```
3. **Build snapshot** (Optional: you can use uploaded snapshot of our teams connections)
   - Change owner names and file paths in snapshot_builder.py to your enriched connections info csv files.
   ```bash
   python -m src.data.snapshot_builder
   ```

4. **Build databases** (one-time setup)
```bash
python -m src.storage.db
python -m src.data.embedding_builder
```

5. **Run the app**
```bash
python -m src.ui.app
```

Open `http://127.0.0.1:5000`

## Architecture

- **Query Understanding**: Extracts structured parameters from natural language
- **Retrieval Strategy**: Decides SQL vs semantic vs hybrid search
- **Synthesis**: Generates natural answers with profile citations and ownership

## Tech Stack

- **Backend**: Flask, LangGraph
- **Storage**: Supabase (Postgres), ChromaDB (vectors)
- **Embeddings**: BAAI/bge-m3
- **LLM**: gpt-5.4-mini (via LiteLLM)

## Team

- Aleksandar Ilić
- Mihajlo Trifunović
- Petar Pavlović
- Mentor: Jelena Graovac
