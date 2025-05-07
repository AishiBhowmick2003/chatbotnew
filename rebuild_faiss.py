
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


PDF_FILE = r"C:\Users\bhowm\OneDrive\Documents\DeployNewChatbot-main\BrainTumorGuidev12.1.pdf"  # Replace with your PDF file name
DB_FAISS_PATH = os.path.abspath("./chatbot_model/db_faiss")

if not os.path.exists(PDF_FILE):
    raise FileNotFoundError(f" PDF file '{PDF_FILE}' not found!")

loader = PyMuPDFLoader(PDF_FILE)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding_model)

db.save_local(DB_FAISS_PATH)
print("FAISS index rebuilt successfully from the single PDF!")
