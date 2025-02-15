from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# PDF
pdf_loader = DirectoryLoader('data', glob="*.pdf", loader_cls=PyPDFLoader)
pdf_documents = pdf_loader.load()

# KSON
json_loader = JSONLoader(
    file_path="./data/courses_info.json",  
    jq_schema='map("Name: \\(.name)\\nFaculty: \\(.faculty)\\nStart Time: \\(.start_time)\\nDays: \\(.days)\\nEnrolled: \\(.std_enrolled)\\nClass Limit: \\(.class_limit)\\nClass Code: \\(.class_code)")[]',
    text_content=True  
)
json_documents = json_loader.load()

documents = pdf_documents + json_documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_db = FAISS.from_documents(texts, embeddings)

faiss_db.save_local("ppc_vector_db")
