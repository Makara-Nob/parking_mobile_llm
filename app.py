import os
import shutil
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

# LangChain & Transformers
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN") # Get token from environment
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "chroma_db"
DATA_DIRECTORY = "data"
os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

app = FastAPI(title="Parking LLM Microservice")

# --- GLOBAL VARIABLES ---
qa_chain = None
vector_store = None

# --- INITIALIZATION ---
def init_model():
    global qa_chain, vector_store
    
    # 1. Embeddings
    print("Loading Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    # 2. Vector DB
    print("Loading Vector DB...")
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # 3. LLM (API vs Local)
    if HF_TOKEN:
        print("🌍 Using Hugging Face Serverless API (Free Tier Mode)")
        llm = HuggingFaceEndpoint(
            repo_id=MODEL_NAME,
            task="text-generation",
            max_new_tokens=512,
            top_k=50,
            temperature=0.1,
            repetition_penalty=1.1,
            huggingfacehub_api_token=HF_TOKEN
        )
    else:
        print(f"🖥️ Using Local GPU Mode: {MODEL_NAME}...")
        if not torch.cuda.is_available():
            print("⚠️ WARNING: No GPU detected! This will be very slow.")
            
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=256,
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # 4. Chain
    prompt_template = """
    <|system|>
    You are a helpful AI assistant for the Smart Parking App. Answer strictly based on the context provided.
    Context: {context}</s>
    <|user|>
    {question}</s>
    <|assistant|>
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("✅ System Ready!")

# --- API MODELS ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --- ENDPOINTS ---
@app.on_event("startup")
async def startup_event():
    init_model()

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not qa_chain:
        raise HTTPException(503, "Model not loaded yet")
    
    try:
        result = qa_chain.invoke({"query": request.query})
        sources = [doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])]
        return {"answer": result["result"], "sources": sources}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global qa_chain, vector_store
    
    file_path = os.path.join(DATA_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
