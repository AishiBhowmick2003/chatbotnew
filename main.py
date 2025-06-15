import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Config
PDF_FILE = "./BrainTumorGuidev12.1.pdf"
DB_FAISS_PATH = "./chatbot_model/db_faiss"
HUGGINGFACE_API_KEY = "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"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not set.")

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build or load FAISS index
def build_faiss_index():
    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(f"PDF file '{PDF_FILE}' not found!")
    
    loader = PyMuPDFLoader(PDF_FILE)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("FAISS index rebuilt from PDF.")
    return db

if os.path.exists(DB_FAISS_PATH):
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("FAISS index loaded from disk.")
    except Exception as e:
        print(f"Failed to load FAISS index: {e}. Rebuilding...")
        db = build_faiss_index()
else:
    print("FAISS index not found. Rebuilding...")
    db = build_faiss_index()

# Load LLM from Hugging Face Hub using LangChain
llm = HuggingFaceHub(
    repo_id=HUGGINGFACE_REPO_ID,
    model_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 200,
        "top_k": 50,
        "do_sample": True,
        "return_full_text": False
    },
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

# Setup memory and chain
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    memory=memory,
)

# On-topic filter
def is_on_topic(query: str) -> bool:
    keywords = [
        "hi","hello","explain","details","name","What do you do?","research","brain",
        "mortality","risk","symptoms","brain tumor", "glioblastoma", "MRI", "surgery",
        "radiation", "tumor", "brain cancer", "treatment", "biopsy", "grading", "scan",
        "astrocytoma", "headache", "vision", "seizure", "neurology", "oncology", "cancer"
    ]
    return any(keyword in query.lower() for keyword in keywords)

# Request body model
class QueryRequest(BaseModel):
    query: str

# Chat endpoint
@app.post("/chat")
async def chat(request: QueryRequest):
    query = request.query.strip()

    if not is_on_topic(query):
        raise HTTPException(status_code=400, detail="Query is off-topic. Please ask about brain tumor-related topics.")

    try:
        result = qa_chain.invoke({"question": query})
        answer = result["answer"]
        clean_answer = answer.split("Unhelpful Answer:")[0].strip()
        return {"response": clean_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
