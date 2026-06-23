import warnings
import os
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

SUPPORTED_EXTENSIONS = [
    ".py",      # Python
    ".java",    # Java
    ".js",      # JavaScript
    ".ts",      # TypeScript
    ".jsx",     # React
    ".tsx",     # React TypeScript
    ".html",    # HTML
    ".css",     # CSS
    ".cpp",     # C++
    ".c",       # C
    ".go",      # Go
    ".rs",      # Rust
]

SKIP_DIRS = [
    "node_modules", ".git", "venv", "__pycache__", 
    "build", "dist", "target", ".idea", ".vscode"
]

def index_codebase(codebase_path: str):
    # Step 1: Load all supported files
    documents = []
    for root, dirs, files in os.walk(codebase_path):
        # Skip junk folders
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for file in files:
            if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                filepath = os.path.join(root, file)
                try:
                    loader = TextLoader(filepath, encoding="utf-8")
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Skipped {filepath}: {e}")

    print(f"Loaded {len(documents)} files")

    if len(documents) == 0:
        print("No supported files found. Check your path.")
        return None

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Step 3: Embed and store in ChromaDB
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("Codebase indexed and saved to chroma_db!")
    return vectorstore

if __name__ == "__main__":
    index_codebase("TEST")