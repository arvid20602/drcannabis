#%% Load env
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

#%% Streamlit UI
st.title("🌿 Dr Cannabis")
st.write("Ask factual questions about cannabis strains, effects, and medical use.")

#%% LLM
MODEL_NAME = "openai/gpt-oss-20b"
llm = ChatGroq(
    model=MODEL_NAME,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

#%% Output schema
class CannabisOutput(BaseModel):
    strain_name: str
    type: str = Field(description="Indica, Sativa, or Hybrid")
    thc_percent: float
    cbd_percent: float
    effects: list[str]
    medical_uses: list[str]
    warnings: list[str]

parser = PydanticOutputParser(pydantic_object=CannabisOutput)

#%% Prompt
# prompt
messages = [
    ("system", "You are a cannabis expert. Use schema {format_instructions}"),
    ("user", "{question}")
]

prompt_template = ChatPromptTemplate.from_messages(messages).partial(
    format_instructions=parser.get_format_instructions()
)

# model
model = ChatGroq(model="openai/gpt-oss-20b")

#%% Chain
chain = prompt_template | model | parser

#%% Chat input
user_input = st.chat_input("Ask about a cannabis strain...")

if user_input:
    with st.spinner("Thinking..."):
        try:
            result = chain.invoke({"question": user_input})
            st.json(result.model_dump())
        except Exception as e:
            st.error(f"Error: {e}")

#%%