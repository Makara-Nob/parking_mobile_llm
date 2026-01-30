import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.core.config import HF_TOKEN, MODEL_NAME, EMBEDDING_MODEL, PERSIST_DIRECTORY, DATA_DIRECTORY
from src.core.constants import SYSTEM_RULES, USER_PROMPT_TEMPLATE

class RAGService:
    def __init__(self):
        self.rag_chain = None
        self.vector_store = None
        self.retriever = None
        self.embeddings = None

    def initialize(self):
        print("Loading Embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        print("Loading Vector DB...")
        self.vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings
        )
        
        self._auto_index()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        if not HF_TOKEN:
            raise Exception("CRITICAL ERROR: HF_TOKEN environment variable is missing.")

        print("Loading LLM...")
        llm_base = HuggingFaceEndpoint(
            repo_id=MODEL_NAME,
            task="text-generation",
            max_new_tokens=256,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.01,
            stop_sequences=["\n\n", "Question:", "Q:", "Context:"]
        )
        llm = ChatHuggingFace(llm=llm_base)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_RULES),
            ("user", USER_PROMPT_TEMPLATE)
        ])

        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG System Ready!")

    def _format_docs(self, docs):
        context = "\n\n".join(doc.page_content for doc in docs)
        return context

    def _auto_index(self):
        current_data = self.vector_store.get()
        if not current_data or not current_data['ids']:
            print("Empty Vector DB detected. Indexing default knowledge base...")
            kb_path = os.path.join(DATA_DIRECTORY, "smart_parking_knowledge_base.txt")
            if os.path.exists(kb_path):
                loader = TextLoader(kb_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)
                self.vector_store.add_documents(chunks)
                print(f"Indexed {len(chunks)} chunks from {kb_path}")

    def query(self, question: str):
        if not self.rag_chain:
            raise Exception("RAG Chain not initialized")
        return self.rag_chain.invoke(question)

    def add_document(self, chunks):
        self.vector_store.add_documents(chunks)
        self.vector_store.persist()
