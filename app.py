import warnings
import os
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.retriever import get_retriever
from src.indexer import index_codebase
import gradio as gr

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert code assistant. 
    Answer questions about the codebase using the provided code context.
    Be concise, clear, and mention which function/file is relevant.
    If the context doesn't contain enough information, say so honestly."""),
    ("human", """Code context:
{context}

Question: {question}""")
])

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Build RAG chain
def build_chain():
    retriever = get_retriever()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Gradio chat function
chain = build_chain()

def ask_codebase(question, history):
    response = chain.invoke(question)
    return response

# Gradio UI
demo = gr.ChatInterface(
    fn=ask_codebase,
    title="🤖 Codebase Assistant",
    description="Ask anything about your codebase!",
    examples=[
        "How is discount calculated?",
        "How does payment processing work?",
        "What does the send_email function do?"
    ]
)

if __name__ == "__main__":
    print("Starting Codebase Assistant...")
    demo.launch()