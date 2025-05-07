import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
PDF_FILE = "./BrainTumorGuidev12.1.pdf"
DB_FAISS_PATH = "./chatbot_model/db_faiss"
HUGGINGFACE_API_KEY = "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"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not set.")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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
llm = HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_REPO_ID,
    task="text-generation",
    temperature=0.5,
    max_new_tokens=200,
    top_k=50,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
)

memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    memory=memory,
)

def is_on_topic(query: str) -> bool:
    keywords = ["brain tumor", "glioblastoma", "MRI", "surgery", "radiation", "tumor", "brain cancer", "treatment", "biopsy", "grading","explain","detail","elaborate","understand","other symptoms","brain tumor", "brain tumour", "tumor", "tumour", "brain", "mass", "lesion",
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
    "progression free survival", "tumor registry", "cohort study", "immunotherapy pipeline"]
    return any(keyword in query.lower() for keyword in keywords)

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: QueryRequest):
    query = request.query.strip()

    if not is_on_topic(query):
        raise HTTPException(status_code=400, detail="❌ Query is off-topic. Please ask about brain tumor-related topics.")

    try:
        result = qa_chain.invoke({"question": query})
        return {"response": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"💥 Error generating response: {str(e)}")







