import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models.schemas import QueryRequest, QueryResponse
from src.services.rag_service import RAGService
from src.core.config import DATA_DIRECTORY
from src.core.constants import GREETINGS, WELCOME_MESSAGE

router = APIRouter()
rag_service = RAGService()

@router.on_event("startup")
async def startup_event():
    rag_service.initialize()

@router.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    raw_query = request.query
    query = raw_query.strip().lower()
    
    # Simple greeting check
    clean_query = "".join(c for c in query if c.isalnum() or c.isspace()).strip()
    if not clean_query or clean_query in GREETINGS:
        return {"answer": WELCOME_MESSAGE}
        
    try:
        answer = rag_service.query(raw_query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        rag_service.add_document(chunks)
        
        return {"message": f"Successfully processed {file.filename} and added to Knowledge Base"}
    except Exception as e:
        raise HTTPException(500, f"Failed to process file: {str(e)}")
