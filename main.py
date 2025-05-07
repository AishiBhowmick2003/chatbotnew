from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel
import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

HUGGINGFACE_API_KEY = "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"
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in your environment variables.")

class QueryRequest(BaseModel):
    query: str

DB_FAISS_PATH = "./db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = None
qa_chain = None

if os.path.exists(DB_FAISS_PATH):
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully!")
    except Exception as e:
        print(f"⚠️ Failed to load FAISS index: {e}")
else:
    print("⚠️ FAISS index not found. Skipping vector store loading for now.")

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_REPO_ID,
    task="text-generation",
    temperature=0.5,
    max_new_tokens=200,
    top_k=50,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
)
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
if db:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        memory=memory,
    )

def is_on_topic(query: str) -> bool:
    keywords = [
    "explain","detail","elaborate","understand","other symptoms","brain tumor", "brain tumour", "tumor", "tumour", "brain", "mass", "lesion",
    "glioblastoma", "astrocytoma", "oligodendroglioma", "meningioma", "ependymoma",
    "medulloblastoma", "schwannoma", "craniopharyngioma", "pituitary adenoma",
    "choroid plexus tumor", "pineoblastoma", "glioma", "metastatic tumor",
    "MRI", "CT", "PET", "neuroimaging", "imaging", "scan", "contrast enhancement",
    "diffusion", "FLAIR", "T1", "T2", "ADC", "perfusion", "edema", "biopsy",
    "histopathology", "grading", "classification", "segmentation", "radiomics",
    "headache", "nausea", "vomiting", "blurred vision", "double vision", "seizure",
    "memory loss", "confusion", "balance issues", "speech difficulty", "cognitive decline",
    "personality change", "motor dysfunction",
    "surgery", "neurosurgery", "craniotomy", "radiotherapy", "radiation", "chemotherapy",
    "temozolomide", "bevacizumab", "targeted therapy", "immunotherapy", "gamma knife",
    "proton therapy", "tumor treating fields", "palliative care",
    "recurrence", "tumor progression", "surveillance", "follow-up", "clinical trials",
    "Karnofsky score", "neurological exam", "steroids", "dexamethasone",
    "neuro-oncology", "oncology", "neurology", "neuroscience", "neuro surgeon",
    "brain cancer", "CNS tumor", "central nervous system", "malignant", "benign",
    "IDH mutation", "MGMT methylation", "1p/19q co-deletion", "EGFR", "TP53", "ATRX",
    "molecular markers", "biomarkers", "genomic profiling", "gliosis", "necrosis",
    "proliferation index", "Ki-67", "BRAF mutation", "H3K27M mutation",
    "TERT promoter mutation", "next-gen sequencing", "oncogene", "tumor suppressor",
    "frontal lobe", "temporal lobe", "parietal lobe", "occipital lobe", "cerebellum",
    "brainstem", "pituitary gland", "pineal gland", "ventricles", "corpus callosum",
    "radiation planning", "brain mapping", "stereotactic surgery", "neuronavigation",
    "intraoperative MRI", "biomarker testing", "tumor board", "clinical pathway",
    "spectroscopy", "functional MRI", "tractography", "susceptibility weighted imaging",
    "MRS", "infiltration", "mass effect", "midline shift",
    "occupational therapy", "speech therapy", "cognitive rehab", "psychosocial support",
    "neuropsychological evaluation", "palliative team", "fatigue management",
    "phase 1 trial", "placebo controlled", "double blind", "overall survival",
    "progression free survival", "tumor registry", "cohort study", "immunotherapy pipeline"
    ]
    query = query.lower()
    return any(keyword in query for keyword in keywords)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_FILE = "./BrainTumorGuidev12.1.pdf"

def build_faiss_index():
    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(f"PDF file '{PDF_FILE}' not found!")

    loader = PyMuPDFLoader(PDF_FILE)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(DB_FAISS_PATH)
    return db
if os.path.exists(DB_FAISS_PATH):
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        db = build_faiss_index()
else:
    print("FAISS index not found. Rebuilding...")
    db = build_faiss_index()

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        user_query = request.query.strip().lower()

        if user_query in ["hi", "hello", "hey", "start"]:
            return {"response": "Hello! 👋 I'm here to help with brain tumor-related questions. How can I assist you today?"}

        if qa_chain is None:
            return {"response": "Vector store not loaded. Please upload documents or load the FAISS index first."}

        if not is_on_topic(user_query):
            return {"response": "I'm here to help with brain tumor-related topics. Could you please ask a question related to that?"}

        response = qa_chain.invoke({"question": request.query})
        return {"response": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/load_index")
def load_faiss_index():
    global db, qa_chain
    if os.path.exists(DB_FAISS_PATH):
        try:
            db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=db.as_retriever(search_kwargs={'k': 2}),
                memory=memory,
            )
            return {"message": "FAISS index loaded and QA chain initialized!"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load FAISS index: {e}")
    else:
        raise HTTPException(status_code=404, detail="FAISS index not found.")

@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running!"}

# Run server
PORT = int(os.environ.get("PORT", 10000))  
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)






