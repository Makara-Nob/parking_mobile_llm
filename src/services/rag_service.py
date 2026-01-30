import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from upstash_vector import Index
from langchain_community.vectorstores import UpstashVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.core.config import HF_TOKEN, MODEL_NAME, EMBEDDING_MODEL, DATA_DIRECTORY, UPSTASH_VECTOR_REST_URL, UPSTASH_VECTOR_REST_TOKEN
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

        print("Connecting to Upstash Vector DB...")
        index = Index(url=UPSTASH_VECTOR_REST_URL, token=UPSTASH_VECTOR_REST_TOKEN)
        self.vector_store = UpstashVectorStore(
            index=index,
            embedding=self.embeddings
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
        # Check if index is empty. Upstash doesn't have a direct .get() for all IDs easily.
        # We can use the .info() method if available, or just a simple similarity search check.
        try:
            # Try to get index info to check vector count
            info = self.vector_store.index.info()
            vector_count = info.vector_count
        except:
            # Fallback: search for something generic
            results = self.vector_store.similarity_search("the", k=1)
            vector_count = len(results)

        if vector_count == 0:
            print("Empty Upstash Vector DB detected. Indexing default knowledge base...")
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

    def get_indexed_sources(self):
        # Upstash integration doesn't support a simple .get() in the same way Chroma does
        # for listing all metadata easily. For now, we'll keep it as a placeholder
        # or research Upstash specific listing if necessary.
        return ["Source listing not supported on Upstash yet"]
