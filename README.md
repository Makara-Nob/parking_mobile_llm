---
title: Parking Llm
emoji: 🏎️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Smart Parking LLM Microservice

This project is a high-performance **Smart Parking AI Assistant**—a microservice that uses Retrieval-Augmented Generation (RAG) to answer real-time questions based on a dedicated knowledge base. 

Rather than relying purely on generalized AI, it securely queries a domain knowledge base to ensure that all answers are highly accurate, factual, and strictly relevant to parking operations and policies.

## 🚀 Key Features

- **Intelligent Q&A (`POST /chat`)**: Send a question, and the API semantically searches the knowledge base, fetches relevant policies/instructions, and synthesizes a natural, accurate response.
- **Dynamic Knowledge Base (`POST /upload`)**: Upload new documents (`.pdf`, `.txt`) instantly. The API automatically chunks, embeds, and indexes these new files into the vector database on the fly.
- **Smart Initialization**: If deployed from scratch, the system automatically detects an empty database and self-indexes the baseline `smart_parking_knowledge_base.txt`.

## 🛠️ The Tech Stack

- **Core Framework:** **FastAPI** for handling high-throughput asynchronous API requests with lightning speed.
- **AI Orchestration:** **LangChain** powers the RAG pipeline.
- **Large Language Model (LLM):** **Llama-3.2-3B-Instruct** (via Hugging Face Serverless Endpoints), optimized for precise conversational tasks.
- **Embeddings Engine:** **Local `all-MiniLM-L6-v2`** (Sentence Transformers) efficiently converts text data into dense semantic vectors.
- **Memory & Storage:** **Upstash Vector DB**, a serverless vector database providing low-latency similarity search.

## 📁 Project Structure

```text
parking-llm/
├── src/
│   ├── api/          # API Route definitions
│   ├── core/         # Config and Constants
│   ├── models/       # Pydantic Schemas
│   ├── services/     # Business Logic (RAG Service)
│   └── main.py       # Application Entry Point
├── data/             # Knowledge Base files (.txt, .pdf)
├── Dockerfile        # Container configuration
└── ARCHITECTURE.md   # Detailed Architecture & Tech Stack
```

## 💻 Running Locally (VS Code)

### Prerequisites
- Python 3.9+
- A [Hugging Face](https://huggingface.co/) Token (`HF_TOKEN`)
- An [Upstash Vector](https://upstash.com/) Database URL & Token

### Installation & Setup
1. **Clone the repository** and open the folder in **VS Code**.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables**:
   Create a `.env` file or export them directly in your VS Code terminal:
   - Command Prompt:
     ```cmd
     set HF_TOKEN=your_hf_token_here
     set UPSTASH_VECTOR_REST_URL=your_upstash_url_here
     set UPSTASH_VECTOR_REST_TOKEN=your_upstash_token_here
     ```
   - PowerShell:
     ```powershell
     $env:HF_TOKEN="your_hf_token_here"
     $env:UPSTASH_VECTOR_REST_URL="your_upstash_url_here"
     $env:UPSTASH_VECTOR_REST_TOKEN="your_upstash_token_here"
     ```
4. **Run the application**:
   Open a terminal in VS Code and start the FastAPI server:
   ```bash
   python -m src.main
   ```
5. **Access the API**:
   You can explore and test the interactive API documentation by navigating to:
   [http://localhost:8000/docs](http://localhost:8000/docs)
