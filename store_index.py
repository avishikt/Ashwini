from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load API keys from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Safety check
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Please set it in .env")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in .env")

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ---- Load and preprocess documents ----
extracted_data = load_pdf_file(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# ---- Load embeddings ----
embeddings = download_hugging_face_embeddings()  # should return a LangChain Embeddings object

# ---- Initialize Pinecone ----
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Check if index exists
indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in indexes:
    pc.create_index(
        name=index_name,
        dimension=384,   # must match your embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Get the index object
index = pc.Index(index_name)

# ---- Store documents ----
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
    namespace="default"  # optional, but good practice
)

print("✅ Documents successfully stored in Pinecone index:", index_name)
