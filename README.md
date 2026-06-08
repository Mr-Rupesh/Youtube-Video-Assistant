# YouTube Video Assistant

A conversational AI app that lets you chat with any YouTube video. Paste a video ID, and the app builds a searchable knowledge base from the transcript — then answers your questions with timestamp-referenced responses.

---

## Overview

This project is a full RAG (Retrieval-Augmented Generation) pipeline built on top of YouTube transcripts. It goes beyond basic similarity search by combining query rewriting, MMR retrieval, and cross-encoder reranking to deliver accurate, context-aware answers — all within a multi-turn conversational interface.

**Key highlights:**

- Fetches YouTube transcripts via Supadata API (handles cloud IP blocks that break direct transcript fetching)
- Embeds transcript chunks into a FAISS vector store for fast semantic search
- Rewrites user queries using a second LLM (Zephyr-7b) before retrieval — improving search accuracy
- Reranks retrieved chunks with a cross-encoder (`ms-marco-MiniLM-L-6-v2`) to surface the most relevant passages
- Uses MMR (Maximal Marginal Relevance) retrieval to reduce redundancy in results
- Responses include `[MM:SS]` timestamps pointing to where in the video the answer comes from
- Maintains full conversation memory across turns with context window trimming
- Smart routing — detects whether the user is asking about the video or just having a general chat

---

## Tech Stack

- **Framework:** Streamlit
- **LLM:** DeepSeek-V3 via HuggingFace Endpoint
- **Query Rewriter:** Zephyr-7b-beta
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** FAISS
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Transcript Source:** Supadata API
- **RAG Orchestration:** LangChain (core, community, HuggingFace, text-splitters)

---

## How It Works

```
YouTube Video ID
      ↓
Supadata API → raw transcript with timestamps
      ↓
RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
      ↓
HuggingFace Embeddings → FAISS vector store
      ↓
User question → Query Rewriter (Zephyr-7b)
      ↓
MMR Retrieval (top 6 of 12 candidates)
      ↓
Cross-Encoder Reranking → top 3 chunks
      ↓
Prompt with video context + conversation history
      ↓
DeepSeek-V3 → answer with [MM:SS] timestamps
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Mr-Rupesh/Youtube-Video-Assistant.git
cd Youtube-Video-Assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure secrets

Create a `.streamlit/secrets.toml` file:

```toml
HUGGINGFACEHUB_API_TOKEN = "your-huggingface-token"
SUPADATA_API_KEY = "your-supadata-key"
```

- Get a HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Get a Supadata key at [supadata.ai](https://supadata.ai)

### 4. Run the app

```bash
streamlit run youtube_chat.py
```

### 5. Usage

1. Find the video ID from any YouTube URL — it's the part after `v=` (e.g. `dQw4w9WgXcQ`)
2. Paste it into the sidebar and the app will load the transcript
3. Ask questions about the video — or just chat freely

---

## Contact

**Rupesh Malhipparge** — [rupeshmalhipparge@gmail.com](mailto:rupeshmalhipparge@gmail.com)
