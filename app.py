import os
import shutil
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

# LangChain & Transformers
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "chroma_db"
DATA_DIRECTORY = "data"
os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

app = FastAPI(title="Parking LLM Microservice")

# --- GLOBAL VARIABLES ---
rag_chain = None
vector_store = None
retriever = None

def format_docs(docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    print(f"\n--- DEBUG: Retrieved Context (len {len(docs)}) ---\n{context}\n----------------------------------\n")
    return context

# --- INITIALIZATION ---
def init_model():
    global rag_chain, vector_store, retriever
    
    print("Loading Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    print("Loading Vector DB...")
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    
    # --- AUTO-INDEXING ---
    current_data = vector_store.get()
    if not current_data or not current_data['ids']:
        print("Empty Vector DB detected. Indexing default knowledge base...")
        kb_path = os.path.join(DATA_DIRECTORY, "smart_parking_knowledge_base.txt")
        if os.path.exists(kb_path):
            loader = TextLoader(kb_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            vector_store.add_documents(chunks)
            print(f"Indexed {len(chunks)} chunks from {kb_path}")
        else:
            print(f"WARNING: Knowledge base not found at {kb_path}")

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    if HF_TOKEN:
        print("Using Hugging Face Serverless API (Chat Mode)")
        llm_base = HuggingFaceEndpoint(
            repo_id=MODEL_NAME,
            task="text-generation",
            max_new_tokens=128,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.01,
            stop_sequences=["\n\n", "Question:", "Q:", "Context:"]
        )
        llm = ChatHuggingFace(llm=llm_base)
    else:
        raise Exception("CRITICAL ERROR: HF_TOKEN environment variable is missing. Please add it to Space Secrets.")

    # 4. RAG Chain using ChatPromptTemplate
    system_rules = (
        "You are the official assistant for the Smart Parking Mobile App. "
        "Your only job is to provide factual answers based ONLY on the provided context.\n\n"
        "STRICT CONSTRAINTS:\n"
        "- Do NOT infer information. If the exact term or topic (e.g., 'refund') is not mentioned, you MUST refuse.\n"
        "- If information is missing, say EXACTLY: 'I don’t have that information.'\n"
        "- Do NOT apologize. Do NOT add prefixes like 'A:' or 'Answer:'.\n"
        "- Output the answer directly and nothing else."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_rules),
        ("user", "CONTEXT:\n{context}\n\nQUESTION: {question}\n\n(Follow STRICT CONSTRAINTS. No inference. If missing, say 'I don’t have that information.')")
    ])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("System Ready!")

# --- API MODELS ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# --- ENDPOINTS ---
@app.on_event("startup")
async def startup_event():
    init_model()

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(503, "Model not loaded yet")
    
    raw_query = request.query
    query = raw_query.strip().lower()
    
    # Remove basic punctuation for greeting check
    clean_query = "".join(c for c in query if c.isalnum() or c.isspace()).strip()
    greetings = {"hi", "hello", "hey", "greeting", "hello there", "halo", "morning", "afternoon", "evening"}
    
    print(f"\n--- DEBUG: Received Query: '{raw_query}' (clean: '{clean_query}') ---")
    
    if not clean_query or clean_query in greetings:
        response = "Hello! I am the Smart Parking Assistant. How can I help you today?"
        print(f"--- DEBUG: Returning Greeting Response ---")
        return {"answer": response}
        
    try:
        print(f"--- DEBUG: Invoking RAG Chain ---")
        answer = rag_chain.invoke(raw_query)
        return {"answer": answer}
    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(500, str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_store
    
    # Ensure data directory exists
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    
    file_path = os.path.join(DATA_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process File
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        # Add to DB
        vector_store.add_documents(chunks)
        vector_store.persist()
        
        return {"message": f"Successfully processed {file.filename} and added to Knowledge Base"}
    except Exception as e:
        print(f"Error in upload: {e}")
        raise HTTPException(500, f"Failed to process file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

