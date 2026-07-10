import warnings
import os
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from src.retriever import search_codebase_tool
import gradio as gr

load_dotenv()

# Initialize LLM

from langchain_openai import ChatOpenAI

from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(
    model="z-ai/glm-5.2",
    api_key=os.getenv("NVIDIA_API_KEY"),
    max_tokens=1024
)
# Tools list
tools = [search_codebase_tool]

# Create ReAct agent using LangGraph
from langchain_core.messages import SystemMessage

system_prompt = """You are a Codebase Assistant. You help developers understand code.

You have access to ONE tool: search_codebase — use it to find relevant code.

For questions that don't need code search (like math, general questions), answer directly without using any tool.
Only use search_codebase when the question is about the codebase."""

agent = create_react_agent(
    llm, 
    tools,
    prompt=system_prompt
)

# Gradio function
import time

def ask_codebase(question, history):
    messages = []
    for msg in history:
        if isinstance(msg, dict):
            messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            human, assistant = msg
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
    
    messages.append({"role": "user", "content": question})
    
    for attempt in range(3):
        try:
            response = agent.invoke({"messages": messages})
            return response["messages"][-1].content
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait_time = (attempt + 1) * 10  # 10s, then 20s
                time.sleep(wait_time)
                continue
            elif "429" in str(e):
                return "Rate limit hit. Please wait a few seconds and try again."
            return f"Error: {str(e)}"

# Gradio UI
demo = gr.ChatInterface(
    fn=ask_codebase,
    title="🤖 Codebase Assistant Agent",
    description="Ask anything about your codebase!",
    examples=[
        "What does this project do?",
        "What machine learning models are used?",
        "How does SHAP explainability work?",
        "Are there any bugs in the code?"
    ]
)

if __name__ == "__main__":
    print("Starting Codebase Assistant Agent...")
    demo.launch()