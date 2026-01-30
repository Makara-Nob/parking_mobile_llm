---
title: Parking Llm
emoji: 🏎️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# RAG System with Free Hugging Face Models

This folder contains a Jupyter Notebook (`rag_pipeline_demo.ipynb`) designed to be run in Google Colab. It sets up a complete Retrieval-Augmented Generation (RAG) system using free,## 📁 Project Structure

```text
parking-llm/
├── src/
│   ├── api/          # API Route definitions
│   ├── core/         # Config and Constants
│   ├── models/        # Pydantic Schemas
│   ├── services/      # Business Logic (RAG Service)
│   └── main.py       # Application Entry Point
├── data/             # Knowledge Base files (.txt, .pdf)
├── chroma_db/        # Persisted vector database
├── Dockerfile        # Container configuration
└── ARCHITECTURE.md    # Detailed Architecture & Tech Stack
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Hugging Face Token (HF_TOKEN)

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variable: `export HF_TOKEN=your_token_here`
4. Run the application: `python -m src.main`
   - Remove excessive headers/footers if possible.
   - Ensure encoding is UTF-8.
   - For PDFs, ensure they are text-searchable (not just scanned images), or the loader will need OCR (which is slower and requires extra dependencies like `tesseract`).

**Example Data Structure:**
```
/content/
  └── data/
       ├── company_policy.pdf
       ├── meeting_notes.txt
       └── technical_manual.pdf
```

## 2. Using the Notebook

1. **Upload Notebook**: Upload `rag_pipeline_demo.ipynb` to Google Colab.
2. **Runtime**: Go to `Runtime` > `Change runtime type` > Select **T4 GPU** (Required for the model to run reasonably fast).
3. **Run Cells**: Execute the cells in order.
   - **Step 1**: Installs dependencies.
   - **Step 2**: Loads the Model (`Zephyr-7B` or `Mistral-7B`).
   - **Step 3**: Defines the Data Pipeline (Loading -> Splitting -> Embedding -> Indexing).
   - **Step 4**: Interactively Chat with your data.

## 3. The Model

We used `HuggingFaceH4/zephyr-7b-beta`.
- **License**: MIT (Free for usage).
- **Size**: ~7 Billion parameters.
- **Quantization**: We use 4-bit quantization (via `bitsandbytes`) so it fits easily into the free Colab T4 GPU (which has ~16GB VRAM).

## 4. The Pipeline Automations
The notebook automates:
- **Chunking**: Breaks large documents into smaller implementation pieces (e.g., 1000 characters) to fit into context chunks.
- **Embedding**: Converts text to vectors using `sentence-transformers/all-MiniLM-L6-v2` (high speed, good performance).
- **Retrieval**: Finds the most relevant chunks for your question.
