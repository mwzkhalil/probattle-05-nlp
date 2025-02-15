# from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS

# loader = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader)
# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
# texts = text_splitter.split_documents(documents)

# embedings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Creates vector embeddings and saves it in the FAISS DB
# faiss_db = FAISS.from_documents(texts, embedings)

# # Saves and export the vector embeddings databse
# faiss_db.save_local("ppc_vector_db")

# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS

# # Load PDF documents
# pdf_loader = DirectoryLoader('data', glob="*.pdf", loader_cls=PyPDFLoader)
# pdf_documents = pdf_loader.load()

# # Load JSON documents
# json_loader = JSONLoader(
#     file_path="./data/courses_info.json",  
#     jq_schema=".messages[].content",
#     text_content=False
# )
# json_documents = json_loader.load()

# documents = pdf_documents + json_documents


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
# texts = text_splitter.split_documents(documents)

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Create FAISS vector store
# faiss_db = FAISS.from_documents(texts, embeddings)

# # Save FAISS database
# faiss_db.save_local("ppc_vector_db")





from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load PDF documents from the 'data' directory
pdf_loader = DirectoryLoader('data', glob="*.pdf", loader_cls=PyPDFLoader)
pdf_documents = pdf_loader.load()

# Load JSON documents using a jq schema that avoids the pipe operator
json_loader = JSONLoader(
    file_path="./data/courses_info.json",  # Adjust the path as needed
    jq_schema='map("Name: \\(.name)\\nFaculty: \\(.faculty)\\nStart Time: \\(.start_time)\\nDays: \\(.days)\\nEnrolled: \\(.std_enrolled)\\nClass Limit: \\(.class_limit)\\nClass Code: \\(.class_code)")[]',
    text_content=True  # Ensure the output is treated as text
)
json_documents = json_loader.load()

# Combine PDF and JSON documents
documents = pdf_documents + json_documents

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create embeddings using a sentence-transformer model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store from the embedded documents
faiss_db = FAISS.from_documents(texts, embeddings)

# Save the FAISS vector store locally
faiss_db.save_local("ppc_vector_db")
