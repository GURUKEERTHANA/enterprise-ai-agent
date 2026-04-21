# src/itsm_agent/agent/llms.py
import os
from langchain_openai import ChatOpenAI

router_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
)

synthesizer_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
)
