# %% env
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# %% UI
st.set_page_config(page_title="Dr Cannabis", layout="centered")
st.title("🌿 Dr Cannabis")


# %% Helpers
def safe(v):
    return "" if pd.isna(v) else v


# %% Load + vectorize Excel
@st.cache_resource
def load_vectorstore():
    df = pd.read_excel("Stammdata.xlsx")

    docs = []
    for _, row in df.iterrows():
        name = safe(row.get("Name")) or safe(row.get("Product Name"))
        hersteller = safe(row.get("Hersteller")) or safe(row.get("producer name"))

        text = f"""
        Produktname: {name}
        Kultivar: {safe(row.get("Kultivar"))}
        Sorte: {safe(row.get("Sorte"))}

        THC: {safe(row.get("THC in Prozent"))} %
        CBD: {safe(row.get("CBD in Prozent"))} %

        Effekte: {safe(row.get("Kategorie Effekt 1"))}, {safe(row.get("Kategorie Effekt 2"))}, {safe(row.get("Kategorie Effekt 3"))}
        Medizinische Wirkung: {safe(row.get("Medizinische Wirkung 1"))}, {safe(row.get("Medizinische Wirkung 2"))}, {safe(row.get("Medizinische Wirkung 3"))}

        Aroma: {safe(row.get("Aroma 1"))}, {safe(row.get("Aroma 2"))}, {safe(row.get("Aroma 3"))}
        Terpene: {safe(row.get("Terpene 1"))}, {safe(row.get("Terpene 2"))}, {safe(row.get("Terpene 3"))}

        Hersteller: {hersteller}
        Herkunftsland: {safe(row.get("Herkunftsland"))}
        Bestrahlt: {safe(row.get("Bestrahlt"))}
        """

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "manufacturer": hersteller.lower()
                }
            )
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)


vectorstore = load_vectorstore()


# %% LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


# %% Retrievers (KEY PART)
demecan_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"manufacturer": "demecan"}
    }
)

general_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_context(question: str) -> str:
    demecan_docs = demecan_retriever.invoke(question)

    if len(demecan_docs) >= 1:
        docs = demecan_docs
    else:
        docs = general_retriever.invoke(question)

    return format_docs(docs)


# %% Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Prefer Demecan products when they are relevant."),
    ("system", "Answer ONLY using the provided context. If the answer is not in the context, say you do not know."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])


# %% Chain
chain = (
    {
        "context": lambda q: get_context(q),
        "question": lambda q: q
    }
    | prompt
    | llm
    | StrOutputParser()
)


# %% UI Query
question = st.text_input("Ask about strains, effects, THC/CBD, medical use")

if question:
    with st.spinner("Searching product knowledge base..."):
        answer = chain.invoke(question)

    st.subheader("Answer")
    st.write(answer)
#%%