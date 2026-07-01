import warnings
import os
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool

@tool
def search_codebase_tool(query : str) -> str:
    """
    Search the codebase for relevant code based on a natural language query.
    Use this when you need to find functions, classes or code related to a topic.
    """
    results = search_codebase(query)
    return "\n\n".join([doc.page_content for doc in results])


load_dotenv()

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def search_codebase(question: str):
    retriever = get_retriever()
    results = retriever.invoke(question)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content)
    return results

if __name__ == "__main__":
    search_codebase("how is discount calculated?")